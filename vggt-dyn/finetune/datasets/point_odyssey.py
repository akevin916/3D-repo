import os
from typing import List

import cv2
import numpy as np
import torch

from .common import (
    crop_resize_to_fixed,
    load_rgb_np,
    make_clip_aug_params,
    make_intrinsics,
    must_exist,
    sliding_windows,
)


class PointOdysseyClipDataset(torch.utils.data.Dataset):
    """PointOdyssey clip dataset with world-to-camera extrinsics from anno.npz."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 8,
        stride: int = 4,
        max_depth: float = 1000.0,
        target_hw: tuple = (294, 518),
        aug_crop: int = 0,
    ):
        base = os.path.join(root, split)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"PointOdyssey split folder not found: {base}")

        self.max_depth = max_depth
        self.target_hw = target_hw
        self.aug_crop = aug_crop
        self.samples: List[dict] = []

        seqs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        for seq in seqs:
            seq_dir = os.path.join(base, seq)
            anno_path = os.path.join(seq_dir, "anno.npz")
            must_exist(anno_path, "Missing anno.npz")
            anno = np.load(anno_path, allow_pickle=True)
            intrinsics = anno["intrinsics"].astype(np.float32)
            extrinsics = anno["extrinsics"].astype(np.float32)[:, :3, :]

            rgb_dir = os.path.join(seq_dir, "rgbs")
            dpt_dir = os.path.join(seq_dir, "depths")
            rgbs = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".jpg"))
            dpts = sorted(f for f in os.listdir(dpt_dir) if f.endswith(".png"))
            n = min(len(rgbs), len(dpts), intrinsics.shape[0], extrinsics.shape[0])
            for win in sliding_windows(n, clip_len, stride):
                self.samples.append(
                    {
                        "rgb_dir": rgb_dir,
                        "dpt_dir": dpt_dir,
                        "rgbs": rgbs,
                        "dpts": dpts,
                        "intrinsics": intrinsics,
                        "extrinsics": extrinsics,
                        "idxs": win,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        imgs, depths, Ks, Es = [], [], [], []

        aug = make_clip_aug_params(self.aug_crop)
        for i in s["idxs"]:
            img = load_rgb_np(os.path.join(s["rgb_dir"], s["rgbs"][i]))
            depth16 = cv2.imread(os.path.join(s["dpt_dir"], s["dpts"][i]), cv2.IMREAD_ANYDEPTH)
            depth = depth16.astype(np.float32) / 65535.0 * 1000.0
            depth = np.clip(depth, 0, self.max_depth)
            K = s["intrinsics"][i]

            img, depth, K = crop_resize_to_fixed(img, depth, K, self.target_hw, aug)

            imgs.append(torch.from_numpy(img.transpose(2, 0, 1).copy()))
            depths.append(torch.from_numpy(depth.copy()))
            Ks.append(torch.from_numpy(K))
            Es.append(torch.from_numpy(s["extrinsics"][i]))

        S, H, W = len(imgs), depths[0].shape[0], depths[0].shape[1]
        return {
            "images": torch.stack(imgs),
            "depths": torch.stack(depths),
            "intrinsics": torch.stack(Ks),
            "extrinsics": torch.stack(Es),
            "dynamic_mask": torch.zeros(S, H, W, dtype=torch.bool),
        }
