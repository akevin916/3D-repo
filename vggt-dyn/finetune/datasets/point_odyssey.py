import os
from typing import List

import cv2
import numpy as np
import torch

from .common import load_rgb, make_intrinsics, must_exist, sliding_windows


class PointOdysseyClipDataset(torch.utils.data.Dataset):
    """PointOdyssey clip dataset with world-to-camera extrinsics from anno.npz."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 8,
        stride: int = 4,
        max_depth: float = 1000.0,
    ):
        base = os.path.join(root, split)
        if not os.path.isdir(base):
            alt = os.path.join(root, "sample")
            if os.path.isdir(alt):
                base = alt
            else:
                raise FileNotFoundError(f"PointOdyssey split folder not found: {base}")

        self.max_depth = max_depth
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

        for i in s["idxs"]:
            imgs.append(load_rgb(os.path.join(s["rgb_dir"], s["rgbs"][i])))
            depth16 = cv2.imread(os.path.join(s["dpt_dir"], s["dpts"][i]), cv2.IMREAD_ANYDEPTH)
            depth = depth16.astype(np.float32) / 65535.0 * 1000.0
            depth = np.clip(depth, 0, self.max_depth)
            depths.append(torch.from_numpy(depth))
            Ks.append(torch.from_numpy(s["intrinsics"][i]))
            Es.append(torch.from_numpy(s["extrinsics"][i]))

        S, H, W = len(imgs), depths[0].shape[0], depths[0].shape[1]
        return {
            "images": torch.stack(imgs),
            "depths": torch.stack(depths),
            "intrinsics": torch.stack(Ks),
            "extrinsics": torch.stack(Es),
            "dynamic_mask": torch.zeros(S, H, W, dtype=torch.bool),
        }
