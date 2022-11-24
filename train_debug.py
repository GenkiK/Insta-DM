"""
PyTorch version 1.4.0, 1.7.0 confirmed

RUN SCRIPT:
./scripts/train_resnet_256_kt.sh
./scripts/train_resnet_256_cs.sh

"""

import argparse
import datetime
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from matplotlib import pyplot as plt

import custom_transforms
import models
from datasets.my_sequence_folders import SequenceFolderWithEgoPose
from loss_functions import (
    compute_mof_consistency_loss,
    compute_obj_size_constraint_loss,
    compute_photo_and_geometry_loss,
    compute_smooth_loss,
)
from rigid_warp import forward_warp, mat2euler

warnings.simplefilter("ignore", UserWarning)


parser = argparse.ArgumentParser(
    description="Learning Monocular Depth in Dynamic Scenes via Instance-Aware Projection Consistency (KITTI and CityScapes)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("data_dir", metavar="DIR", type=Path, help="path to dataset", default="")
parser.add_argument("--sequence-length", type=int, metavar="N", help="sequence length for training", default=3)
parser.add_argument("-mni", type=int, help="maximum number of instances", default=20)
parser.add_argument(
    "--rotation-mode",
    type=str,
    choices=["euler", "quat"],
    default="euler",
    help="rotation mode for PoseExpnet : euler (yaw, pitch, roll) or quaternion (last 3 coefficients)",
)
parser.add_argument(
    "--padding-mode",
    type=str,
    choices=["zeros", "border"],
    default="zeros",
    help="padding mode for image warping : this is important for photometric differenciation when going outside target image."
    " zeros will null gradients outside target image."
    " border will only null gradients of the coordinate outside (x or y)",
)


parser.add_argument("-j", "--workers", default=1, type=int, metavar="N", help="number of data loading workers")
parser.add_argument("-b", "--batch-size", default=2, type=int, metavar="N", help="mini-batch size")
parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument(
    "--epoch-size", default=0, type=int, metavar="N", help="manual epoch size (will match dataset size if not set)"
)
parser.add_argument(
    "--disp-lr",
    "--disp-learning-rate",
    default=1e-4,
    type=float,
    metavar="LR",
    help="initial learning rate for DispResNet",
)
parser.add_argument(
    "--ego-lr",
    "--ego-learning-rate",
    default=1e-4,
    type=float,
    metavar="LR",
    help="initial learning rate for EgoPoseNet",
)
parser.add_argument(
    "--obj-lr",
    "--obj-learning-rate",
    default=1e-4,
    type=float,
    metavar="LR",
    help="initial learning rate for ObjPoseNet",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum for sgd, alpha parameter for adam"
)
parser.add_argument("--beta", default=0.999, type=float, metavar="M", help="beta parameters for adam")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, metavar="W", help="weight decay")
parser.add_argument(
    "--resnet-layers", type=int, default=18, choices=[18, 50], help="number of ResNet layers for depth estimation."
)
parser.add_argument("--with-pretrained", type=int, default=1, help="with or without imagenet pretrained for resnet")
parser.add_argument("--resnet-pretrained", action="store_true", help="pretrained from resnet model or not")
parser.add_argument("--seed", default=0, type=int, help="seed for random functions, and network initialization")

parser.add_argument(
    "-p", "--photo-loss-weight", type=float, help="weight for photometric loss", metavar="W", default=2.0
)
parser.add_argument(
    "-c",
    "--geometry-consistency-weight",
    type=float,
    help="weight for depth consistency loss",
    metavar="W",
    default=1.0,
)
parser.add_argument(
    "-s", "--smooth-loss-weight", type=float, help="weight for disparity smoothness loss", metavar="W", default=0.1
)
parser.add_argument(
    "-o", "--scale-loss-weight", type=float, help="weight for object scale loss", metavar="W", default=0.02
)
parser.add_argument(
    "-mc", "--mof-consistency-loss-weight", type=float, help="weight for mof consistency loss", metavar="W", default=0.1
)
parser.add_argument("--translation-weight", type=float, help="weight for ego pose loss", metavar="W", default=0.5)
parser.add_argument("--rotation-weight", type=float, help="weight for ego pose loss", metavar="W", default=0.5)

parser.add_argument("--with-auto-mask", action="store_true", help="with the the mask for stationary points")
parser.add_argument("--with-ssim", action="store_true", help="with ssim or not")
parser.add_argument(
    "--with-mask", action="store_true", help="with the the mask for moving objects and occlusions or not"
)
parser.add_argument("--with-only-obj", action="store_true", help="with only obj mask")

parser.add_argument("--debug-mode", action="store_true", help="run codes with debugging mode or not")
parser.add_argument("--no-shuffle", action="store_true", help="feed data without shuffling")
parser.add_argument("--no-input-aug", action="store_true", help="feed data without augmentation")
parser.add_argument("--begin-idx", type=int, default=None, help="beginning index for pre-processed data")


best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device_val = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    print(
        "=> PyTorch version: " + torch.__version__ + " || CUDA_VISIBLE_DEVICES: " + os.environ["CUDA_VISIBLE_DEVICES"]
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Data loading
    normalize = custom_transforms.NormalizeWithEgoPose(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_transform = custom_transforms.ComposeWithEgoPose([custom_transforms.ArrayToTensorWithEgoPose(), normalize])

    print("=> fetching scenes from '{}'".format(args.data_dir))
    train_set = SequenceFolderWithEgoPose(
        root_dir=args.data_dir,
        is_train=True,
        seed=args.seed,
        shuffle=True,
        max_num_instances=args.mni,
        transform=train_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrained).to(device)
    ego_pose_net = models.EgoPoseNet(18, args.with_pretrained).to(device)
    obj_pose_net = models.ObjPoseNet(18, args.with_pretrained).to(device)

    ego_pose_net.init_weights()
    obj_pose_net.init_weights()
    disp_net.init_weights()

    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    ego_pose_net = torch.nn.DataParallel(ego_pose_net)
    obj_pose_net = torch.nn.DataParallel(obj_pose_net)

    print("=> setting adam solver")

    optim_params = []
    if args.disp_lr != 0:
        optim_params.append({"params": disp_net.module.encoder.parameters(), "lr": args.disp_lr})
        optim_params.append({"params": disp_net.module.decoder.parameters(), "lr": args.disp_lr})
        optim_params.append({"params": disp_net.module.obj_height_prior, "lr": args.disp_lr * 0.1})
    if args.ego_lr != 0:
        optim_params.append({"params": ego_pose_net.parameters(), "lr": args.ego_lr})
    if args.obj_lr != 0:
        optim_params.append({"params": obj_pose_net.parameters(), "lr": args.obj_lr})

    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    for _ in range(args.epochs):
        ### train for one epoch ###
        train(args, train_loader, disp_net, ego_pose_net, obj_pose_net, optimizer, args.epoch_size)


def train(args, train_loader, disp_net, ego_pose_net, obj_pose_net, optimizer, epoch_size):
    global n_iter
    torch.set_printoptions(sci_mode=False)
    np.set_printoptions(suppress=True)

    w1, w2, w3 = args.photo_loss_weight, args.geometry_consistency_weight, args.smooth_loss_weight
    w4, w5 = args.scale_loss_weight, args.mof_consistency_loss_weight
    translation_weight, rotation_weight = args.translation_weight, args.rotation_weight
    # loss_1: photometric (eq.17)
    # loss_2: geometric (eq.18)
    # loss_3: smooth (eq.19)
    # loss_4: height loss (eq. 21)
    # loss_5: translation constraint (eq. 20)
    # loss_6: x
    # loss_7: x

    # switch to train mode
    disp_net.train()
    ego_pose_net.train()
    obj_pose_net.train()

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, tgt_insts, ref_insts, gt_rel_poses) in enumerate(
        train_loader
    ):

        ### inputs to GPU ###
        tgt_img = tgt_img.to(device)
        ref_imgs = [ref_img.to(device) for ref_img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)
        tgt_insts = [tgt_inst.to(device) for tgt_inst in tgt_insts]
        ref_insts = [
            ref_inst.to(device) for ref_inst in ref_insts
        ]  # ref_insts[i] == ref_imgs[i]とtgt_imgと比較したときにassociationできたref_imgs[i]内のインスタンスたち
        gt_rel_poses = [pose.to(device) for pose in gt_rel_poses]
        breakpoint()

        ### input instance masking ###
        tgt_bg_masks = [
            1 - (inst_img[:, 1:].sum(dim=1, keepdim=True) > 0).float() for inst_img in tgt_insts
        ]  # inst_img: torch.Size([max_num_insts, H, W])
        ref_bg_masks = [1 - (inst_img[:, 1:].sum(dim=1, keepdim=True) > 0).float() for inst_img in ref_insts]
        tgt_bg_imgs = [tgt_img * tgt_mask * ref_mask for tgt_mask, ref_mask in zip(tgt_bg_masks, ref_bg_masks)]
        ref_bg_imgs = [
            ref_img * tgt_mask * ref_mask for ref_img, tgt_mask, ref_mask in zip(ref_imgs, tgt_bg_masks, ref_bg_masks)
        ]
        tgt_obj_masks = [1 - mask for mask in tgt_bg_masks]
        ref_obj_masks = [1 - mask for mask in ref_bg_masks]
        num_insts = [
            tgt_inst[:, 0, 0, 0].int().detach().cpu().numpy().tolist() for tgt_inst in tgt_insts
        ]  # Number of instances for each sequence
        breakpoint()

        ### object height prior ###
        height_prior = disp_net.module.obj_height_prior

        ### compute depth & ego-motion ###
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)

        # def compute_ego_pose_with_inv_and_loss(ego_pose_net, tgt_bg_imgs, ref_bg_imgs, gt_rel_poses):
        ego_poses_fwd = []
        ego_poses_bwd = []
        translation_loss, rotation_loss = (
            torch.tensor(0.0, requires_grad=True).cuda(),
            torch.tensor(0.0, requires_grad=True).cuda(),
        )
        for tgt_img, ref_img, batch_rel_poses in zip(tgt_bg_imgs, ref_bg_imgs, gt_rel_poses):
            # gt_pose: [r11, r12, r13, t1, r21, r22, t2, r31, r32, r33, t3].view(3, 4)
            gt_translation_vecs = batch_rel_poses[:, :, 3]
            gt_rot_mats = batch_rel_poses[:, :, :3]
            print(f"gt_trans_vec:\n{gt_translation_vecs}")
            # print(f"gt_trans_vec_bwd:\n{- gt_translation_vecs.unsqueeze(1) @ gt_rot_mats}")
            # print(f"gt_rot_mat:\n{gt_rot_mats}")
            # print(f"gt_rot_mat_bwd:\n{gt_rot_mats.transpose(1, 2)}")
            print("")

            fwd_pose = ego_pose_net(tgt_img, ref_img)
            bwd_pose = ego_pose_net(ref_img, tgt_img)
            # if i % 100 == 0:
            # print(fwd_pose)
            # print(bwd_pose)
            # print("")

            # fwd_pose.retain_grad()
            # bwd_pose.retain_grad()
            ego_poses_fwd.append(fwd_pose)
            ego_poses_bwd.append(bwd_pose)

            fwd_rotation_loss = torch.linalg.norm(fwd_pose[:, 3:] - mat2euler(gt_rot_mats), dim=1).mean()
            # 並進ベクトルのL2Normのロス
            fwd_translation_loss = torch.linalg.norm(gt_translation_vecs - fwd_pose[:, :3], dim=1).mean()
            # fwd_rotation_loss.retain_grad()
            # fwd_translation_loss.retain_grad()

            bwd_rotation_loss = torch.linalg.norm(
                bwd_pose[:, 3:] - mat2euler(gt_rot_mats.transpose(1, 2)), dim=1
            ).mean()
            bwd_translation_loss = torch.linalg.norm(
                bwd_pose[:, :3] - (-gt_translation_vecs.view(-1, 1, 3) @ gt_rot_mats), dim=1
            ).mean()  # 並進ベクトルのL2Normのロス
            # bwd_rotation_loss.retain_grad()
            # bwd_translation_loss.retain_grad()

            # print("\nfwd_translation_loss")
            # print(fwd_translation_loss)
            # print("\nbwd_translation_loss")
            # print(bwd_translation_loss)
            # print("")

            translation_loss = translation_loss + fwd_translation_loss + bwd_translation_loss
            rotation_loss = rotation_loss + fwd_rotation_loss + bwd_rotation_loss

        # translation_loss.retain_grad()
        # rotation_loss.retain_grad()

        ### Remove ego-motion effect: transformation with ego-motion ###
        ### NumRefs(2) >> Nx(C+mni)xHxW,  {t-1}->{t} | {t+1}->{t} ###
        ref2tgt_imgs_ego, ref2tgt_insts_ego, ref2tgt_vals_ego = compute_ego_warp(
            ref_imgs, ref_insts, ref_depths, ego_poses_bwd, intrinsics
        )
        ### NumRefs(2) >> Nx(C+mni)xHxW,  {t}->{t-1} | {t}->{t+1} ###
        tgt2ref_imgs_ego, tgt2ref_insts_ego, tgt2ref_vals_ego = compute_ego_warp(
            [tgt_img, tgt_img], tgt_insts, [tgt_depth, tgt_depth], ego_poses_fwd, intrinsics
        )

        ### Compute object motion ###
        obj_poses_fwd, obj_poses_bwd = compute_obj_pose_with_inv(
            obj_pose_net,
            tgt_img,
            tgt_insts,
            ref2tgt_imgs_ego,
            ref2tgt_insts_ego,
            ref_imgs,
            ref_insts,
            tgt2ref_imgs_ego,
            tgt2ref_insts_ego,
            args.mni,
            num_insts,
        )

        ### Compute composite motion field ###
        mofs_fwd, mofs_bwd = compute_motion_field(
            tgt_img, ego_poses_fwd, ego_poses_bwd, obj_poses_fwd, obj_poses_bwd, tgt_insts, ref_insts
        )

        ### Compute unified projection loss ###
        (
            loss_1,
            loss_2,
            _,
            _,
            ref2tgt_flows,
            tgt2ref_flows,
            ref2tgt_diffs,
            tgt2ref_diffs,
            ref2tgt_valids,
            tgt2ref_valids,
        ) = compute_photo_and_geometry_loss(
            tgt_img,
            ref_imgs,
            intrinsics,
            tgt_depth,
            ref_depths,
            mofs_fwd,
            mofs_bwd,
            args.with_ssim,
            args.with_mask,
            args.with_auto_mask,
            args.padding_mode,
            args.with_only_obj,
            tgt_obj_masks,
            ref_obj_masks,
            ref2tgt_vals_ego,
            tgt2ref_vals_ego,
        )
        ### Compute depth smoothness loss ###
        if w3 == 0:
            loss_3 = torch.tensor(0.0).cuda()
        else:
            loss_3 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        ### Compute object size constraint loss ###
        if w4 == 0:
            loss_4 = torch.tensor(0.0).cuda()
        else:
            loss_4 = compute_obj_size_constraint_loss(
                height_prior, tgt_depth, tgt_insts, ref_depths, ref_insts, intrinsics, args.mni, num_insts
            )

        ### Compute unified motion consistency loss ###
        if w5 == 0:
            loss_5 = torch.tensor(0.0).cuda()
        else:
            loss_5 = compute_mof_consistency_loss(
                mofs_fwd,
                mofs_bwd,
                ref2tgt_flows,
                tgt2ref_flows,
                ref2tgt_diffs,
                tgt2ref_diffs,
                ref2tgt_valids,
                tgt2ref_valids,
                alpha=5,
                thresh=0.1,
            )

        ### Compute height prior constraint loss ###
        # 学習には使ってない．height-priorをTensorboardで見たかっただけ？
        # loss_6 = height_prior
        ### Compute depth mean constraint loss ###
        # 学習には使ってない．これもTensorboardで見たかっただけ？
        # loss_7 = ((1 / tgt_depth).mean() + sum([(1 / depth).mean() for depth in ref_depths])) / (1 + len(ref_depths))

        loss = (
            w1 * loss_1
            + w2 * loss_2
            + w3 * loss_3
            + w4 * loss_4
            + w5 * loss_5
            + translation_weight * translation_loss
            + rotation_weight * rotation_loss
        )
        """
            loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item(), loss_5.item()
            w1*loss_1.item(), w2*loss_2.item(), w3*loss_3.item(), w4*loss_4.item(), w5*loss_5.item()

            -b 4 -p 1.0 -c 0.5 -s 0.05 -o 0.01 -mc 0.01 -hp 0 -dm 0 -mni 3 \
        """

        ### compute gradient and do Adam step ###
        if loss > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(list(ego_pose_net.parameters())[-1])

        if i >= epoch_size - 1:
            break
        n_iter += 1


########################################################################################################################################


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = 1 / disp_net(tgt_img)
    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = 1 / disp_net(ref_img)
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_ego_pose_with_inv(pose_net, tgt_imgs, ref_imgs):
    poses_fwd = []
    poses_bwd = []
    for tgt_img, ref_img in zip(tgt_imgs, ref_imgs):
        poses_fwd.append(pose_net(tgt_img, ref_img))
        poses_bwd.append(pose_net(ref_img, tgt_img))

    return poses_fwd, poses_bwd


def compute_ego_warp(imgs, insts, depths, poses, intrinsics):
    """
    Args:
        imgs:       [[B, 3, 256, 832], [B, 3, 256, 832]]
        insts:      [[B, 3, 256, 832], [B, 3, 256, 832]]
        depths:     [[B, 1, 256, 832], [B, 1, 256, 832]]
        poses:      [[B, 6], [B, 6]]
        intrinsics: [B, 3, 3]
    Returns:
        warped_imgs:     [[B, 3, 256, 832], [B, 3, 256, 832]]
        warped_vals:     [[B, 1, 256, 832], [B, 1, 256, 832]]
    """
    warped_imgs, warped_insts, warped_depths, warped_vals = [], [], [], []
    for img, inst, depth, pose in zip(imgs, insts, depths, poses):
        img_cat = torch.cat([img, inst[:, 1:]], dim=1)
        warped_img_cat, warped_depth, warped_val = forward_warp(
            img_cat, depth.detach(), pose.detach(), intrinsics, upscale=3
        )
        warped_imgs.append(warped_img_cat[:, :3])
        warped_insts.append(torch.cat([inst[:, :1], warped_img_cat[:, 3:].round()], dim=1))
        warped_depths.append(warped_depth)
        warped_vals.append(warped_val)
    return warped_imgs, warped_insts, warped_vals


def compute_obj_pose_with_inv(
    pose_net,
    tgt_img,
    tgt_insts,
    ref2tgt_imgs,
    ref2tgt_insts,
    ref_imgs,
    ref_insts,
    tgt2ref_imgs,
    tgt2ref_insts,
    mni,
    num_insts,
):
    """
    Args:
        ------------------------------------------------
        tgtI:  [B, 3, 256, 832]
        tgtMs: [[B, 1+N, 256, 832], [B, 1+N, 256, 832]]
        r2tIs: [[B, 3, 256, 832], [B, 3, 256, 832]]
        r2tMs: [[B, 1+N, 256, 832], [B, 1+N, 256, 832]]
        ------------------------------------------------
        refIs: [[B, 3, 256, 832], [B, 3, 256, 832]]
        refMs: [[B, 1+N, 256, 832], [B, 1+N, 256, 832]]
        t2rIs: [[B, 3, 256, 832], [B, 3, 256, 832]]
        t2rMs: [[B, 1+N, 256, 832], [B, 1+N, 256, 832]]
        ------------------------------------------------
        intrinsics: [B, 3, 3]
        num_insts:  [[n1, n2, ...], [n1', n2', ...]]
    Returns:
        "Only translations (tx, ty, tz) are estimated!"
        obj_poses_fwd: [[B, N, 3], [B, N, 3]]
        obj_poses_bwd: [[B, N, 3], [B, N, 3]]
    """
    batch_size, _, h, w = tgt_img.size()

    obj_poses_fwd, obj_poses_bwd = [], []

    for tgt_inst, ref2tgt_img, ref2tgt_inst, ref_img, ref_inst, tgt2ref_img, tgt2ref_inst, num_inst in zip(
        tgt_insts, ref2tgt_imgs, ref2tgt_insts, ref_imgs, ref_insts, tgt2ref_imgs, tgt2ref_insts, num_insts
    ):
        obj_pose_fwd = torch.zeros([batch_size * mni, 3], dtype=tgt_img.dtype)
        obj_pose_bwd = torch.zeros([batch_size * mni, 3], dtype=tgt_img.dtype)

        if sum(num_inst) != 0:
            tgt_img_repeated = tgt_img.repeat_interleave(mni, dim=0)
            tgt_inst_repeated = tgt_inst[:, 1:].reshape(-1, 1, h, w)
            fwd_indices = (
                tgt_inst_repeated.mean(dim=[1, 2, 3]) != 0
            ).detach()  # tgt, judge each channel whether instance exists
            tgtO = (tgt_img_repeated * tgt_inst_repeated)[fwd_indices]

            ref2tgt_img_repeated = ref2tgt_img.repeat_interleave(mni, dim=0)
            ref2tgt_inst_repeated = ref2tgt_inst[:, 1:].reshape(-1, 1, h, w)
            ref2tgtO = (ref2tgt_img_repeated * ref2tgt_inst_repeated)[fwd_indices]

            ref_img_repeated = ref_img.repeat_interleave(mni, dim=0)
            ref_inst_repeated = ref_inst[:, 1:].reshape(-1, 1, h, w)
            bwd_idx = (
                ref_inst_repeated.mean(dim=[1, 2, 3]) != 0
            ).detach()  # ref, judge each channel whether instance exists
            refO = (ref_img_repeated * ref_inst_repeated)[bwd_idx]

            tgt2ref_img_repeated = tgt2ref_img.repeat_interleave(mni, dim=0)
            tgt2ref_inst_repeated = tgt2ref_inst[:, 1:].reshape(-1, 1, h, w)
            tgt2refO = (tgt2ref_img_repeated * tgt2ref_inst_repeated)[bwd_idx]

            pose_fwd = pose_net(tgtO, ref2tgtO)
            pose_bwd = pose_net(refO, tgt2refO)
            obj_pose_fwd[fwd_indices] = pose_fwd
            obj_pose_bwd[bwd_idx] = pose_bwd

        obj_poses_fwd.append(obj_pose_fwd.reshape(batch_size, mni, 3))
        obj_poses_bwd.append(obj_pose_bwd.reshape(batch_size, mni, 3))

    return obj_poses_fwd, obj_poses_bwd


def compute_motion_field(tgt_img, ego_poses_fwd, ego_poses_bwd, obj_poses_fwd, obj_poses_bwd, tgt_insts, ref_insts):
    """
    Args:
        ego_poses_fwd: [torch.Size([B, 6]), torch.Size([B, 6])]
        ego_poses_bwd: [torch.Size([B, 6]), torch.Size([B, 6])]
        obj_poses_fwd: [torch.Size([B, N, 6]), torch.Size([B, N, 6])]
        obj_poses_bwd: [torch.Size([B, N, 6]), torch.Size([B, N, 6])]
        tgt_insts: [torch.Size([B, 1+N, 256, 832]), torch.Size([B, 1+N, 256, 832])]
        ref_insts: [torch.Size([B, 1+N, 256, 832]), torch.Size([B, 1+N, 256, 832])]
    Returns:
        motion_fields_fwd: [ ([B, 6, 256, 832]), ([B, 6, 256, 832]) ]
        motion_fields_bwd: [ ([B, 6, 256, 832]), ([B, 6, 256, 832]) ]
    """
    batch_size, _, h, w = tgt_img.size()
    motion_fields_fwd, motion_fields_bwd = [], []  # [ ([B, 6, 256, 832]), ([B, 6, 256, 832]) ]

    for ego_pose_fwd, ego_pose_bwd, obj_pose_fwd, obj_pose_bwd, tgt_inst, ref_inst in zip(
        ego_poses_fwd, ego_poses_bwd, obj_poses_fwd, obj_poses_bwd, tgt_insts, ref_insts
    ):
        if (tgt_inst[:, 1:].sum(dim=1) > 1).sum() + (ref_inst[:, 1:].sum(dim=1) > 1).sum():
            print("WARNING: overlapped instance region at {}".format(datetime.datetime.now().strftime("%m-%d-%H:%M")))

        # [batch_size, 6, h, w]
        motion_field_fwd = ego_pose_fwd.reshape(batch_size, 6, 1, 1).repeat(1, 1, h, w)
        motion_field_bwd = ego_pose_bwd.reshape(batch_size, 6, 1, 1).repeat(1, 1, h, w)

        # おそらくインスタンス領域でマスクかけてる
        # dim2の"3"は並進の次元数(?)(回転は考えていない)
        # [batch_size, mni, 3, h, w]
        obj_motion_field_fwd = tgt_inst[:, 1:].unsqueeze(2) * obj_pose_fwd.unsqueeze(-1).unsqueeze(-1)
        obj_motion_field_bwd = ref_inst[:, 1:].unsqueeze(2) * obj_pose_bwd.unsqueeze(-1).unsqueeze(-1)

        motion_field_fwd[:, :3] += obj_motion_field_fwd.sum(dim=1, keepdim=False)
        motion_field_bwd[:, :3] += obj_motion_field_bwd.sum(dim=1, keepdim=False)
        motion_fields_fwd.append(motion_field_fwd)
        motion_fields_bwd.append(motion_field_bwd)
    return motion_fields_fwd, motion_fields_bwd


def save_image(data, cm, fn, vmin=None, vmax=None):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data, cmap=cm, vmin=vmin, vmax=vmax)
    plt.savefig(fn, dpi=height)
    plt.close()


if __name__ == "__main__":
    main()
