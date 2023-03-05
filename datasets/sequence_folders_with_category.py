"""
(+) added customized outputs: flow_fwd, flow_bwd, segmentation mask (src/tgt), instance mask (src/tgt)
(+) added recursive_check_nonzero_inst
"""
from __future__ import annotations

import math
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.data as data
from imageio import imread

from custom_transforms import Compose
from flow_io import flow_read
from rigid_warp import flow_warp


def load_img_as_float(path: Path):
    return imread(path).astype(np.float32)


def load_flo_as_float(path: Path):
    return np.array(flow_read(path)).astype(np.float32)


def load_seg_and_category_as_tensor(path: Path):
    npz = np.load(path)
    masks = torch.from_numpy(npz["masks"].astype(np.float32))
    labels = torch.from_numpy(npz["labels"].astype(np.int32))
    return masks, labels


def load_invalid_idxs(path: Path) -> list[int]:
    with open(path, "rb") as f:
        return pickle.load(f)


# def load_seg_and_category_as_tensor(path):
#     npz = np.load(path)
#     masks = torch.from_numpy(npz["masks"].astype(np.float32))
#     labels = torch.from_numpy(npz["labels"].astype(np.int32))
#     # 背景を追加
#     # maskがmaskが[[]]みたいなときに死ぬからハードコーディングしているけど修正する．
#     if masks.squeeze().shape[0] == 0:
#         masks = torch.zeros((1, 256, 832), dtype=torch.float32)
#         labels = torch.tensor([-1], dtype=torch.int32)
#     else:
#         masks = torch.cat((torch.zeros_like(masks[0])[None, :, :], masks), dim=0)
#         # 背景のカテゴリーIDは-1
#         labels = torch.cat((torch.tensor([-1], dtype=torch.int32), labels))
#     return masks, labels


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


class SequenceFolder(data.Dataset):
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
        max_n_inst: int = 10,
        transform: Compose | None = None,
        proportion: int = 1,
        begin_idx: int = 0,
    ) -> None:
        np.random.seed(seed)
        random.seed(seed)
        scenes_txt_path = root_dir / "train.txt" if is_train else root_dir / "val.txt"
        with open(scenes_txt_path, "r") as f:
            scene_names = f.read().splitlines()
        self.samples = SequenceFolder.make_samples(root_dir, scene_names, shuffle)
        split_index = int(math.floor(len(self.samples) * proportion))
        self.samples = self.samples[begin_idx:split_index]
        self.mni = max_n_inst
        self.transform = transform

    @staticmethod
    def read_height_priors(root_dir: Path):
        # indexがラベルIDに対応している．0列目がexpectation, 1列目がvariance
        path = root_dir / "height_priors.txt"
        return torch.from_numpy(np.loadtxt(path, delimiter=" ", dtype=np.float32, ndmin=2))

    @staticmethod
    def make_samples(root_dir: Path, scene_names: list[str], shuffle: bool):
        samples: list[dir[str, Any]] = []
        outlier_idxs_root_dir = root_dir / "outlier_indices"
        for scene_name in scene_names:
            scene_img_dir = root_dir / "image" / scene_name
            scene_flof_dir = root_dir / "flow_f" / scene_name
            scene_flob_dir = root_dir / "flow_b" / scene_name
            scene_segm_dir = root_dir / "segmentation_postprocess" / scene_name
            intrinsics = np.genfromtxt(scene_img_dir / "cam.txt").astype(np.float32).reshape(3, 3)

            img_paths = sorted(scene_img_dir.glob("*.jpg"))
            flof_paths = sorted(scene_flof_dir.glob("*.flo"))  # 00: src, 01: tgt
            flob_paths = sorted(scene_flob_dir.glob("*.flo"))  # 00: tgt, 01: src
            segm_paths = sorted(scene_segm_dir.glob("*.npz"))
            outlier_idxs_paths = sorted(outlier_idxs_root_dir.glob(f"*/{scene_name}"))

            if len(img_paths) < 3:
                continue
            for tgt_idx in range(1, len(img_paths) - 1):
                sequence_sample = {
                    "intrinsics": intrinsics,
                    "tgt": img_paths[tgt_idx],
                    "ref_imgs": [],
                    "flow_fs": [],
                    "flow_bs": [],
                    "tgt_seg": segm_paths[tgt_idx],
                    "ref_segs": [],
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

        tgt_seg, tgt_inst_labels = load_seg_and_category_as_tensor(sample["tgt_seg"])
        # ref_segs, ref_insts_labels = [load_seg_and_category_as_tensor(path) for path in sample["ref_segs"]]
        ref_segs, ref_insts_labels = [], []
        for seg_path in sample["ref_segs"]:
            ref_seg, ref_insts_label = load_seg_and_category_as_tensor(seg_path)
            ref_segs.append(ref_seg)
            ref_insts_labels.append(ref_insts_label)

        tgt_insts = []
        ref_insts = []
        tgt_insts_matched_labels = []
        ref_insts_matched_labels = []
        tgt_sort = torch.cat([torch.zeros(1).long(), tgt_seg.sum(dim=(1, 2)).argsort(descending=True)[:-1]], dim=0)
        tgt_seg = tgt_seg[tgt_sort]
        tgt_inst_labels = tgt_inst_labels[tgt_sort]

        ref_sorts = [
            torch.cat([torch.zeros(1).long(), ref_seg.sum(dim=(1, 2)).argsort(descending=True)[:-1]], dim=0)
            for ref_seg in ref_segs
        ]
        ref_segs = [ref_seg[ref_sort] for ref_seg, ref_sort in zip(ref_segs, ref_sorts)]
        ref_insts_labels = [ref_insts_label[ref_sort] for ref_insts_label, ref_sort in zip(ref_insts_labels, ref_sorts)]
        # ref_sorts = [torch.cat([torch.zeros(1).long(), ref_seg.sum(dim=(1, 2)).argsort(descending=True)[:-1]], dim=0) for ref_seg in ref_segs if ]
        # tgt_seg, ref_segはセグメンテーション領域が大きい順にインスタンスがch方向に並んでる
        # ref_segs_sorted = [ref_seg[ref_sort] for ref_seg, ref_sort in zip(ref_segs_sorted, ref_sorts)]

        for i in range(len(ref_imgs)):
            # この中ではref_imgs[i]の1枚とtgtのみを比較
            noc_f, noc_b = find_noc_masks(flow_fs[i].unsqueeze(0), flow_bs[i].unsqueeze(0))
            if i < len(ref_imgs) / 2:  # first half
                seg0 = ref_segs[i].unsqueeze(0)
                labels0 = ref_insts_labels[i]
                seg1 = tgt_seg.unsqueeze(0)
                labels1 = tgt_inst_labels
            else:  # second half
                seg0 = tgt_seg.unsqueeze(0)
                labels0 = tgt_inst_labels
                seg1 = ref_segs[i].unsqueeze(0)
                labels1 = ref_insts_labels[i]

            seg0w, _ = flow_warp(seg1, flow_fs[i].unsqueeze(0))
            seg1w, _ = flow_warp(seg0, flow_bs[i].unsqueeze(0))
            # warped_seg0_from_seg1, _ = flow_warp(seg1, flow_fs[i].unsqueeze(0))
            # warped_seg1_from_seg0, _ = flow_warp(seg0, flow_bs[i].unsqueeze(0))

            n_inst0 = seg0.shape[1]
            # n_inst0 = min(seg0.shape[1], self.mni)
            # n_inst1 = min(seg1.shape[1], self.mni)
            # min_n_inst = min(n_inst0, n_inst1)

            # Warp seg0 to seg1. Find IoU between seg1w and seg1. Find the maximum corresponded instance in seg1.
            # たぶんch_01はseg0(ref)の各オブジェクトについて最大のIoUをとるseg1(tgt)のインスタンスのidxが並んでる
            iou_01, ch_01 = inst_iou(seg1w, seg1, valid_mask=noc_b)
            iou_10, ch_10 = inst_iou(seg0w, seg0, valid_mask=noc_f)

            seg0_re = torch.zeros(self.mni + 1, seg0.shape[2], seg0.shape[3])  # seg.shape[2:] == [H, W]
            seg1_re = torch.zeros(self.mni + 1, seg1.shape[2], seg1.shape[3])
            labels0_re = -np.ones(self.mni, dtype=np.int32)
            labels1_re = -np.ones(self.mni, dtype=np.int32)
            non_overlap_0 = torch.ones([seg0.shape[2], seg0.shape[3]])
            non_overlap_1 = torch.ones([seg0.shape[2], seg0.shape[3]])

            n_match = 0
            for ch in range(n_inst0):
                condition1 = labels0[ch] == labels1[ch_01[ch]]
                condition2 = (ch == ch_10[ch_01[ch]]) and (iou_01[ch] > 0.5) and (iou_10[ch_01[ch]] > 0.5)
                condition3 = ((seg0[0, ch] * non_overlap_0).max() > 0) and (
                    (seg1[0, ch_01[ch]] * non_overlap_1).max() > 0
                )
                if condition1 and condition2 and condition3 and (n_match < self.mni):  # matching success!
                    n_match += 1
                    # マッチ数がインデックス＝どんどんassociationできたinstance(ch)をappendしていってるイメージ
                    seg0_re[n_match] = seg0[0, ch] * non_overlap_0
                    seg1_re[n_match] = seg1[0, ch_01[ch]] * non_overlap_1
                    labels0_re[n_match - 1] = labels0[ch]
                    labels1_re[n_match - 1] = labels0[ch]
                    non_overlap_0 = non_overlap_0 * (1 - seg0_re[n_match])
                    non_overlap_1 = non_overlap_1 * (1 - seg1_re[n_match])
            seg0_re[0] = n_match
            seg1_re[0] = n_match

            # この時点でlabelsにはbackgroundの要素が含まれていない
            if i < len(ref_imgs) / 2:  # first half
                tgt_insts.append(seg1_re.detach().cpu().numpy().transpose(1, 2, 0))
                ref_insts.append(seg0_re.detach().cpu().numpy().transpose(1, 2, 0))
                tgt_insts_matched_labels.append(labels1_re)
                ref_insts_matched_labels.append(labels0_re)
            else:  # second half
                tgt_insts.append(seg0_re.detach().cpu().numpy().transpose(1, 2, 0))
                ref_insts.append(seg1_re.detach().cpu().numpy().transpose(1, 2, 0))
                tgt_insts_matched_labels.append(labels0_re)
                ref_insts_matched_labels.append(labels1_re)
            # tgt_insts: [[ref_imgs[0]に対してassociationできたtgt内のインスタンス最大20個], [ref_imgs[1]...], ...]
            # tgt_insts: list[torch.Size([H, W, max_n_inst])]
            # ref_insts: [[ref_imgs[0]とtgtを見比べてassociationできたref_imgs[0]内のインスタンス最大20個], [ref_imgs[1]...], ...]
        intrinsics = np.copy(sample["intrinsics"])
        if self.transform is not None:
            imgs, segms, intrinsics = self.transform([tgt_img] + ref_imgs, tgt_insts + ref_insts, intrinsics)
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
            tgt_insts = segms[: int(len(ref_imgs) / 2 + 1)]  # list[torch.Size([max_n_inst, H, W])]
            ref_insts = segms[int(len(ref_imgs) / 2 + 1) :]

        # While passing through RandomScaleCrop(), instances could be flied-out and become zero-mask. -> Need filtering!
        for sq in range(len(ref_imgs)):
            tgt_insts[sq], ref_insts[sq] = recursive_check_nonzero_inst(tgt_insts[sq], ref_insts[sq])

        return (
            tgt_img,
            ref_imgs,
            intrinsics,
            np.linalg.inv(intrinsics),
            tgt_insts,
            ref_insts,
            tgt_insts_matched_labels,
            ref_insts_matched_labels,
        )

    def __len__(self):
        return len(self.samples)


# if __name__ == "__main__":
#     root_path = Path("/home/gkinoshita/humpback/workspace/Insta-DM/kitti_256/outlier_indices/")
#     root_path = root_path.iterdir().__next__()
#     for scene_path in root_path.iterdir():
#         for i, idx_path in enumerate(scene_path.iterdir()):
#             if i < 10:
#                 print(load_invalid_idxs(idx_path))
#             else:
#                 exit()
