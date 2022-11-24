"""
(+) added customized outputs: flow_fwd, flow_bwd, segmentation mask (src/tgt), instance mask (src/tgt)
(+) added recursive_check_nonzero_inst
"""
from __future__ import annotations

import math
import pdb
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data as data
from custom_transforms import ComposeWithEgoPose
from flow_io import flow_read
from imageio import imread
from rigid_warp import flow_warp


def convert_poses_t_1_to_t(poses: list[torch.Tensor]) -> list[torch.Tensor]:
    # poses: list[torch.Size([3, 4])]
    t_1_to_t_poses: list[torch.Tensor] = []
    for t in range(1, len(poses)):
        pose_t_1 = poses[t - 1]
        pose_t = poses[t]
        t_1_to_t_rot = pose_t_1[:, :3].T @ pose_t[:, :3]  # t-1でみたときのある座標をtでみたときの値
        # t_1_to_t_translation = pose_t_1[:, :3].T @ (pose_t[:, 3].view(-1, 1) - pose_t_1[:, 3].view(-1, 1))
        t_1_to_t_translation = pose_t_1[:, :3].T @ (pose_t[:, 3:] - pose_t_1[:, 3:])
        t_1_to_t_poses.append(torch.cat((t_1_to_t_rot, t_1_to_t_translation), dim=1))
    return t_1_to_t_poses


def convert_poses_t_to_t_1(poses: list[torch.Tensor]) -> list[torch.Tensor]:
    t_to_t_1_poses: list[torch.Tensor] = []
    for t in range(1, len(poses)):
        pose_t_1 = poses[t - 1]
        pose_t = poses[t]
        t_to_t_1_rot = pose_t[:, :3].T @ pose_t_1[:, :3]
        # t_to_t_1_translation = pose_t[:, :3].T @ (pose_t_1[:, 3].view(-1, 1) - pose_t[:, 3].view(-1, 1))
        t_to_t_1_translation = pose_t[:, :3].T @ (pose_t_1[:, 3:] - pose_t[:, 3:])
        t_to_t_1_poses.append(torch.cat((t_to_t_1_rot, t_to_t_1_translation), dim=1))
    return t_to_t_1_poses


def load_img_as_float(path):
    return imread(path).astype(np.float32)


def load_flo_as_float(path):
    out = np.array(flow_read(path)).astype(np.float32)
    return out


def load_seg_as_float(path):
    return np.load(path).astype(np.float32)


def L2_norm(x, dim=1, keepdim=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset, dim=dim, keepdim=keepdim)
    return l2_norm


def find_noc_masks(fwd_flow, bwd_flow):
    """
    fwd_flow: torch.size([1, 2, 256, 832])
    bwd_flow: torch.size([1, 2, 256, 832])
    output: torch.size([1, 1, 256, 832]), torch.size([1, 1, 256, 832])

    input shape of flow_warp(): torch.size([bs, 2, 256, 832])
    """
    bwd2fwd_flow, _ = flow_warp(bwd_flow, fwd_flow)
    fwd2bwd_flow, _ = flow_warp(fwd_flow, bwd_flow)

    fwd_flow_diff = torch.abs(bwd2fwd_flow + fwd_flow)
    bwd_flow_diff = torch.abs(fwd2bwd_flow + bwd_flow)

    fwd_consist_bound = torch.max(0.05 * L2_norm(fwd_flow), torch.Tensor([3.0]))
    bwd_consist_bound = torch.max(0.05 * L2_norm(bwd_flow), torch.Tensor([3.0]))

    noc_mask_0 = (L2_norm(fwd_flow_diff) < fwd_consist_bound).type(
        torch.FloatTensor
    )  # noc_mask_tgt, torch.Size([1, 1, 256, 832]), torch.float32
    noc_mask_1 = (L2_norm(bwd_flow_diff) < bwd_consist_bound).type(
        torch.FloatTensor
    )  # noc_mask_src, torch.Size([1, 1, 256, 832]), torch.float32
    return noc_mask_0, noc_mask_1


# def inst_iou(seg_src, seg_tgt, valid_mask):
#     """
#     srcの各インスタンスにつき，tgtの全てのインスタンスとIoUを計算していく
#     -> Which instance of seg_tgt matches instances of seg_src?

#     seg_src: torch.Size([1, n_inst, 256, 832])
#     seg_tgt:  torch.Size([1, n_inst, 256, 832])
#     valid_mask: torch.Size([1, 1, 256, 832])
#     """
#     n_inst_src = seg_src.shape[1]  # チャネル数==インスタンス数
#     n_inst_tgt = seg_tgt.shape[1]
#     seg_src_m = seg_src * valid_mask.repeat(1, n_inst_src, 1, 1)
#     seg_tgt_m = seg_tgt * valid_mask.repeat(1, n_inst_tgt, 1, 1)
#     """
#     plt.figure(1), plt.imshow(seg_src.sum(dim=0).sum(dim=0)), plt.colorbar(), plt.ion(), plt.show()
#     plt.figure(2), plt.imshow(seg_tgt.sum(dim=0).sum(dim=0)),  plt.colorbar(), plt.ion(), plt.show()
#     plt.figure(3), plt.imshow(valid_mask[0,0]),  plt.colorbar(), plt.ion(), plt.show()
#     plt.figure(4), plt.imshow(seg_src_m.sum(dim=0).sum(dim=0)),  plt.colorbar(), plt.ion(), plt.show()
#     """
#     for i in range(n_inst_src):
#         if i == 0:  # background
#             match_table = torch.zeros((1, n_inst_tgt), dtype=torch.float32)
#             continue
#         overlap = (
#             (seg_src_m[:, i].unsqueeze(1).repeat(1, n_inst_tgt, 1, 1) * seg_tgt_m)
#             .clamp(min=0, max=1)
#             .squeeze(0)
#             .sum(dim=(1, 2))
#         )  # H,Wで合計してIoU面積を出す
#         union = (
#             (seg_src_m[:, i].unsqueeze(1).repeat(1, n_inst_tgt, 1, 1) + seg_tgt_m)
#             .clamp(min=0, max=1)
#             .squeeze(0)
#             .sum(dim=(1, 2))
#         )
#         iou_inst = overlap / union
#         match_table = torch.cat((match_table, iou_inst.unsqueeze(0)), dim=0)

#     iou, inst_idx = torch.max(match_table, dim=1)
#     return iou, inst_idx


def inst_iou(seg_src, seg_tgt, valid_mask):
    """
    srcの各インスタンスにつき，tgtの全てのインスタンスとIoUを計算していく
    -> Which instance of seg_tgt matches instances of seg_src?

    seg_src: torch.Size([1, n_inst, 256, 832])
    seg_tgt:  torch.Size([1, n_inst, 256, 832])
    valid_mask: torch.Size([1, 1, 256, 832])
    """
    n_inst_src = seg_src.shape[1]  # チャネル数==インスタンス数
    n_inst_tgt = seg_tgt.shape[1]
    seg_src_m = seg_src * valid_mask.repeat(1, n_inst_src, 1, 1)
    seg_tgt_m = seg_tgt * valid_mask.repeat(1, n_inst_tgt, 1, 1)
    # plt.imshow(seg_src_m[0].sum(dim=0))
    # plt.show()
    match_table = torch.zeros((n_inst_src, n_inst_tgt), dtype=torch.float32)
    for i in range(1, n_inst_src):  # i=0はbackground
        overlap = (
            (seg_src_m[:, i].unsqueeze(1).repeat(1, n_inst_tgt, 1, 1) * seg_tgt_m)
            .clamp(min=0, max=1)
            .squeeze(0)
            .sum(1)
            .sum(1)
        )  # H,Wで合計してIoU面積を出す
        union = (
            (seg_src_m[:, i].unsqueeze(1).repeat(1, n_inst_tgt, 1, 1) + seg_tgt_m)
            .clamp(min=0, max=1)
            .squeeze(0)
            .sum(1)
            .sum(1)
        )
        # print(f"union: {union}")
        # print(f"overlap: {overlap}")

        # if union < 1:
        # iou_inst = 0
        # else:
        iou_inst = overlap / union
        # print(f"iou_inst: {iou_inst}")
        # print("")
        match_table[i] = iou_inst.unsqueeze(0)
    iou, inst_idx = torch.max(match_table, dim=1)
    return iou, inst_idx


def recursive_check_nonzero_inst(tgt_inst, ref_inst):
    assert tgt_inst[0].mean() == ref_inst[0].mean()
    n_inst = int(tgt_inst[0].mean())
    for nn in range(n_inst):
        if tgt_inst[nn + 1].mean() == 0:
            tgt_inst[0] -= 1
            ref_inst[0] -= 1
            if nn + 1 == n_inst:
                tgt_inst[nn + 1 :] = 0
                ref_inst[nn + 1 :] = 0
            else:
                tgt_inst[nn + 1 :] = torch.cat(
                    [tgt_inst[nn + 2 :], torch.zeros(1, tgt_inst.size(1), tgt_inst.size(2))], dim=0
                )  # re-ordering
                ref_inst[nn + 1 :] = torch.cat(
                    [ref_inst[nn + 2 :], torch.zeros(1, ref_inst.size(1), ref_inst.size(2))], dim=0
                )  # re-ordering
            return recursive_check_nonzero_inst(tgt_inst, ref_inst)
        if ref_inst[nn + 1].mean() == 0:
            tgt_inst[0] -= 1
            ref_inst[0] -= 1
            if nn + 1 == n_inst:
                tgt_inst[nn + 1 :] = 0
                ref_inst[nn + 1 :] = 0
            else:
                tgt_inst[nn + 1 :] = torch.cat(
                    [tgt_inst[nn + 2 :], torch.zeros(1, tgt_inst.size(1), tgt_inst.size(2))], dim=0
                )  # re-ordering
                ref_inst[nn + 1 :] = torch.cat(
                    [ref_inst[nn + 2 :], torch.zeros(1, ref_inst.size(1), ref_inst.size(2))], dim=0
                )  # re-ordering
            return recursive_check_nonzero_inst(tgt_inst, ref_inst)
    return tgt_inst, ref_inst


class SequenceFolderWithEgoPose(data.Dataset):
    """
    A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(
        self,
        root_dir: Path,
        is_train: bool,
        seed: int | None = None,
        shuffle: bool = True,
        max_num_instances: int = 20,
        transform: ComposeWithEgoPose | None = None,
        proportion: int = 1,
        begin_idx: int = 0,
    ) -> None:
        np.random.seed(seed)
        random.seed(seed)
        scenes_txt_path = root_dir / "train.txt" if is_train else root_dir / "val.txt"
        scene_names: list[str] = []
        with open(scenes_txt_path, "r") as f:
            # scenes = [self.root / "image" / folder[:-1] for folder in f]
            scene_names = f.read().splitlines()
        self.samples = SequenceFolderWithEgoPose.make_samples(root_dir, scene_names, shuffle)
        self.mni = max_num_instances
        self.transform = transform
        split_index = int(math.floor(len(self.samples) * proportion))
        self.samples = self.samples[begin_idx:split_index]

    @staticmethod
    def make_samples(root_dir: Path, scene_names: list[str], shuffle: bool):
        # 全てのscene_nameをstereo_typeも含めたものに変更（scene_name = {scene_name}_{stereo_type}).val.txt, train.txt指定する際に扱いやすくするため
        samples: list[dir[str, Any]] = []
        for scene_name in scene_names:
            # scene_poses: list[torch.Tensor] = []
            with open(root_dir / "poses" / f"{scene_name[:-2]}.txt") as f:
                # for line in f:
                #     line = line.rstrip("\n")
                #     if len(line) == 0:
                #         continue
                #     scene_poses.append(torch.Tensor([float(elem) for elem in line.split(" ")]).view(3, 4))
                scene_poses = [
                    torch.Tensor([float(elem) for elem in line.rstrip().split(" ")]).view(3, 4)
                    for line in f
                    if line != "\n"
                ]
            scene_poses_t_to_t_1 = convert_poses_t_to_t_1(scene_poses)
            scene_poses_t_1_to_t = convert_poses_t_1_to_t(scene_poses)
            scene_img_dir = root_dir / "image" / scene_name
            scene_flof_dir = root_dir / "flow_f" / scene_name
            scene_flob_dir = root_dir / "flow_b" / scene_name
            scene_segm_dir = root_dir / "segmentation" / scene_name
            intrinsics = np.genfromtxt(scene_img_dir / "cam.txt").astype(np.float32).reshape(3, 3)

            img_paths = sorted(scene_img_dir.glob("*.jpg"))
            flof_paths = sorted(scene_flof_dir.glob("*.flo"))  # 00: src, 01: tgt
            flob_paths = sorted(scene_flob_dir.glob("*.flo"))  # 00: tgt, 01: src
            segm_paths = sorted(scene_segm_dir.glob("*.npy"))

            if len(img_paths) < 3:
                continue
            for tgt_idx in range(1, len(img_paths) - 1):  # for i in range(half_length, len(imgs) - half_length):
                # demo.pyと異なり，隣り合ってる画像同士じゃなくてtgtと近隣のシークエンスについて比較して学習してる．
                # ここで言うシークエンスはkitti-odometry/sequencesのsequenceとは異なる．
                sequence_sample = {
                    "intrinsics": intrinsics,
                    "tgt": img_paths[tgt_idx],
                    "ref_imgs": [],
                    "flow_fs": [],
                    "flow_bs": [],
                    "tgt_seg": segm_paths[tgt_idx],
                    "ref_segs": [],
                    "poses_t_1_to_t": scene_poses_t_1_to_t[tgt_idx - 1 : tgt_idx + 1],  # tgt_idx-1, tgt_idx
                    "poses_t_to_t_1": scene_poses_t_to_t_1[tgt_idx - 1 : tgt_idx + 1],
                }  # ('tgt_insts':[], 'ref_insts':[]) will be processed when getitem() is called
                # ref_imgsはtgt_imgの前後half_sequence_lengthの画像とセグメンテーション
                for j in [-1, 1]:
                    sequence_sample["ref_imgs"].append(img_paths[tgt_idx + j])
                    sequence_sample["ref_segs"].append(segm_paths[tgt_idx + j])
                for j in [-1, 0]:
                    sequence_sample["flow_fs"].append(flof_paths[tgt_idx + j])
                    sequence_sample["flow_bs"].append(flob_paths[tgt_idx + j])
                samples.append(sequence_sample)
        if shuffle:
            random.shuffle(samples)
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tgt_img = load_img_as_float(sample["tgt"])
        ref_imgs = [load_img_as_float(ref_img) for ref_img in sample["ref_imgs"]]  # tgtの前後数フレーム

        flow_fs = [torch.from_numpy(load_flo_as_float(flow_f)) for flow_f in sample["flow_fs"]]
        flow_bs = [torch.from_numpy(load_flo_as_float(flow_b)) for flow_b in sample["flow_bs"]]

        tgt_seg = torch.from_numpy(load_seg_as_float(sample["tgt_seg"]))
        ref_segs = [torch.from_numpy(load_seg_as_float(ref_seg)) for ref_seg in sample["ref_segs"]]

        # 可視化 for debug
        # if len(tgt_seg) > 0:
        #     fig = plt.figure(figsize=(25, 10))
        #     ax1, ax2, ax3 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)
        #     ax1.imshow(tgt_seg.sum(dim=0))
        #     if len(ref_segs[0]) > 0:
        #         ax2.imshow(ref_segs[0].sum(dim=0))
        #     if len(ref_segs[1]) > 0:
        #         ax3.imshow(ref_segs[1].sum(dim=0))
        #     plt.show()

        if len(tgt_seg) > 0:
            tgt_insts = []
            ref_insts = []
            tgt_sort = torch.cat([torch.zeros(1).long(), tgt_seg.sum(dim=(1, 2)).argsort(descending=True)[:-1]], dim=0)
            tgt_seg = tgt_seg[tgt_sort]

            ref_segs_sorted = []
            for ref_seg in ref_segs:
                if len(ref_seg) > 0:
                    ref_sort = torch.cat(
                        [torch.zeros(1).long(), ref_seg.sum(dim=(1, 2)).argsort(descending=True)[:-1]], dim=0
                    )
                    ref_segs_sorted.append(ref_seg[ref_sort])
                else:
                    # plt.title("not exist ref_inst")
                    # plt.imshow(ref_imgs[ref_idx].astype("uint8"))
                    # plt.show()
                    ref_segs_sorted.append(ref_seg)
            # ref_sorts = [torch.cat([torch.zeros(1).long(), ref_seg.sum(dim=(1, 2)).argsort(descending=True)[:-1]], dim=0) for ref_seg in ref_segs if ]
            # tgt_seg, ref_segはセグメンテーション領域が大きい順にインスタンスがch方向に並んでる
            # ref_segs_sorted = [ref_seg[ref_sort] for ref_seg, ref_sort in zip(ref_segs_sorted, ref_sorts)]

            for i in range(len(ref_imgs)):

                if len(ref_segs_sorted[i]) == 0:
                    tgt_insts.append(np.zeros((tgt_img.shape[0], tgt_img.shape[1], self.mni + 1)))
                    ref_insts.append(np.zeros((tgt_img.shape[0], tgt_img.shape[1], self.mni + 1)))
                    continue

                # この中ではref_imgs[i]の1枚とtgtのみを比較
                noc_f, noc_b = find_noc_masks(flow_fs[i].unsqueeze(0), flow_bs[i].unsqueeze(0))
                # plt.imshow(noc_f[0][0])
                # plt.show()
                # plt.imshow(noc_b[0][0])
                # plt.show()

                if i < len(ref_imgs) / 2:  # first half
                    seg0 = ref_segs_sorted[i].unsqueeze(0)
                    seg1 = tgt_seg.unsqueeze(0)
                else:  # second half
                    seg0 = tgt_seg.unsqueeze(0)
                    seg1 = ref_segs_sorted[i].unsqueeze(0)

                seg0w, _ = flow_warp(seg1, flow_fs[i].unsqueeze(0))
                seg1w, _ = flow_warp(seg0, flow_bs[i].unsqueeze(0))
                # warped_seg0_from_seg1, _ = flow_warp(seg1, flow_fs[i].unsqueeze(0))
                # warped_seg1_from_seg0, _ = flow_warp(seg0, flow_bs[i].unsqueeze(0))

                n_inst0 = seg0.shape[1]

                # Warp seg0 to seg1. Find IoU between seg1w and seg1. Find the maximum corresponded instance in seg1.
                # たぶんch_01はseg0(ref)の各オブジェクトについて最大のIoUをとるseg1(tgt)のインスタンスのidxが並んでる
                iou_01, ch_01 = inst_iou(seg1w, seg1, valid_mask=noc_b)
                iou_10, ch_10 = inst_iou(seg0w, seg0, valid_mask=noc_f)

                seg0_re = torch.zeros(self.mni + 1, seg0.shape[2], seg0.shape[3])  # seg.shape[2:] == [H, W]
                seg1_re = torch.zeros(self.mni + 1, seg1.shape[2], seg1.shape[3])
                non_overlap_0 = torch.ones([seg0.shape[2], seg0.shape[3]])
                non_overlap_1 = torch.ones([seg0.shape[2], seg0.shape[3]])

                num_match = 0
                for ch in range(n_inst0):
                    condition1 = (ch == ch_10[ch_01[ch]]) and (iou_01[ch] > 0.5) and (iou_10[ch_01[ch]] > 0.5)
                    condition2 = ((seg0[0, ch] * non_overlap_0).max() > 0) and (
                        (seg1[0, ch_01[ch]] * non_overlap_1).max() > 0
                    )
                    if condition1 and condition2 and (num_match < self.mni):  # matching success!
                        # fig = plt.figure(figsize=(20, 15))
                        # ax1 = fig.add_subplot(1, 2, 1)
                        # ax1.imshow(seg0[0, ch])
                        # ax2 = fig.add_subplot(1, 2, 2)
                        # ax2.imshow(tgt_img.astype("uint8"))
                        # plt.show()
                        num_match += 1
                        # マッチ数がインデックス＝どんどんassociationできたinstance(ch)をappendしていってるイメージ
                        seg0_re[num_match] = seg0[0, ch] * non_overlap_0
                        seg1_re[num_match] = seg1[0, ch_01[ch]] * non_overlap_1
                        non_overlap_0 = non_overlap_0 * (1 - seg0_re[num_match])
                        non_overlap_1 = non_overlap_1 * (1 - seg1_re[num_match])
                seg0_re[0] = num_match
                seg1_re[0] = num_match

                # fig = plt.figure(figsize=(20, 15))
                # ax1 = fig.add_subplot(1, 2, 1)
                # ax1.imshow(seg0_re[1:].sum(dim=0))
                # ax2 = fig.add_subplot(1, 2, 2)
                # ax2.imshow(tgt_img.astype("uint8"))
                # plt.title("matched")
                # plt.show()
                # print(f"num_match: {num_match}")

                if i < len(ref_imgs) / 2:  # first half
                    tgt_insts.append(seg1_re.detach().cpu().numpy().transpose(1, 2, 0))
                    ref_insts.append(seg0_re.detach().cpu().numpy().transpose(1, 2, 0))
                else:  # second half
                    tgt_insts.append(seg0_re.detach().cpu().numpy().transpose(1, 2, 0))
                    ref_insts.append(seg1_re.detach().cpu().numpy().transpose(1, 2, 0))
                # tgt_insts: [[ref_imgs[0]に対してassociationできたtgt内のインスタンス最大20個], [ref_imgs[1]...], ...]
                # tgt_insts: list[torch.Size([H, W, max_num_insts])]
                # ref_insts: [[ref_imgs[0]とtgtを見比べてassociationできたref_imgs[0]内のインスタンス最大20個], [ref_imgs[1]...], ...]

        else:  # tgt_instが存在しなかった場合
            # plt.title("not exist tgt_inst")
            # plt.imshow(tgt_img.astype("uint8"))
            # plt.show()
            tgt_insts = [np.zeros((tgt_img.shape[0], tgt_img.shape[1], self.mni + 1)) for _ in range(len(ref_imgs))]
            ref_insts = [np.zeros((tgt_img.shape[0], tgt_img.shape[1], self.mni + 1)) for _ in range(len(ref_imgs))]
        intrinsics = np.copy(sample["intrinsics"])
        gt_poses_t_1_to_t: list[torch.Tensor] = deepcopy(sample["poses_t_1_to_t"])
        gt_poses_t_to_t_1: list[torch.Tensor] = deepcopy(sample["poses_t_to_t_1"])
        if self.transform is not None:
            imgs, segms, intrinsics, gt_poses_t_1_to_t, gt_poses_t_to_t_1 = self.transform(
                [tgt_img] + ref_imgs, tgt_insts + ref_insts, intrinsics, gt_poses_t_1_to_t, gt_poses_t_to_t_1
            )
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
            tgt_insts = segms[: int(len(ref_imgs) / 2 + 1)]  # list[torch.Size([max_num_insts, H, W])]
            ref_insts = segms[int(len(ref_imgs) / 2 + 1) :]

        # While passing through RandomScaleCrop(), instances could be flied-out and become zero-mask. -> Need filtering!
        for sq in range(len(ref_imgs)):
            tgt_insts[sq], ref_insts[sq] = recursive_check_nonzero_inst(tgt_insts[sq], ref_insts[sq])

        if tgt_insts[0][0].mean() != 0 and tgt_insts[0][int(tgt_insts[0][0].mean())].mean() == 0:
            pdb.set_trace()
        if tgt_insts[1][0].mean() != 0 and tgt_insts[1][int(tgt_insts[1][0].mean())].mean() == 0:
            pdb.set_trace()
        if ref_insts[0][0].mean() != 0 and ref_insts[0][int(ref_insts[0][0].mean())].mean() == 0:
            pdb.set_trace()
        if ref_insts[1][0].mean() != 0 and ref_insts[1][int(ref_insts[1][0].mean())].mean() == 0:
            pdb.set_trace()

        if tgt_insts[0][0].mean() != tgt_insts[0][1:].mean(-1).mean(-1).nonzero().size(0):
            pdb.set_trace()
        if tgt_insts[1][0].mean() != tgt_insts[1][1:].mean(-1).mean(-1).nonzero().size(0):
            pdb.set_trace()
        if ref_insts[0][0].mean() != ref_insts[0][1:].mean(-1).mean(-1).nonzero().size(0):
            pdb.set_trace()
        if ref_insts[1][0].mean() != ref_insts[1][1:].mean(-1).mean(-1).nonzero().size(0):
            pdb.set_trace()
        return (
            tgt_img,
            ref_imgs,
            intrinsics,
            np.linalg.inv(intrinsics),
            tgt_insts,
            ref_insts,
            gt_poses_t_1_to_t,
            gt_poses_t_to_t_1,
        )

    def __len__(self):
        return len(self.samples)
