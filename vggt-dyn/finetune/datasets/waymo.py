import os
from typing import List

import cv2
import numpy as np
import torch

from .common import c2w_to_w2c, crop_resize_to_fixed, load_rgb_np, make_clip_aug_params, sliding_windows


class WaymoClipDataset(torch.utils.data.Dataset):
    """Waymo clip dataset from preprocessed per-segment folders."""

    def __init__(
        self,
        root: str,
        clip_len: int = 8,
        stride: int = 4,
        camera_id: int = 1,
        max_depth: float = 200.0,
        target_hw: tuple = (392, 518),
        aug_crop: int = 0,
    ):
        self.max_depth = max_depth
        self.target_hw = target_hw
        self.aug_crop = aug_crop
        self.samples: List[dict] = []
        camera_suffix = f"_{camera_id}"

        segs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        for seg in segs:
            seg_dir = os.path.join(root, seg)
            frames = sorted(
                f[:-4]
                for f in os.listdir(seg_dir)
                if f.endswith(".jpg") and f[:-4].endswith(camera_suffix)
            )
            if not frames:
                continue

            good = []
            for fr in frames:
                jpg = os.path.join(seg_dir, fr + ".jpg")
                exr = os.path.join(seg_dir, fr + ".exr")
                npz = os.path.join(seg_dir, fr + ".npz")
                if os.path.exists(jpg) and os.path.exists(exr) and os.path.exists(npz):
                    good.append(fr)

            n = len(good)
            for win in sliding_windows(n, clip_len, stride):
                self.samples.append({"seg_dir": seg_dir, "frames": good, "idxs": win})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        imgs, depths, Ks, Es = [], [], [], []
        aug = make_clip_aug_params(self.aug_crop)
        for i in s["idxs"]:
            fr = s["frames"][i]
            img = load_rgb_np(os.path.join(s["seg_dir"], fr + ".jpg"))

            depth = cv2.imread(os.path.join(s["seg_dir"], fr + ".exr"), cv2.IMREAD_UNCHANGED)
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth = np.clip(depth.astype(np.float32), 0, self.max_depth)

            cam = np.load(os.path.join(s["seg_dir"], fr + ".npz"))
            K = cam["intrinsics"].astype(np.float32)
            c2w = cam["cam2world"].astype(np.float32)

            img, depth, K = crop_resize_to_fixed(img, depth, K, self.target_hw, aug)

            imgs.append(torch.from_numpy(img.transpose(2, 0, 1).copy()))
            depths.append(torch.from_numpy(depth.copy()))
            Ks.append(torch.from_numpy(K))
            Es.append(torch.from_numpy(c2w_to_w2c(c2w)))

        S, H, W = len(imgs), depths[0].shape[0], depths[0].shape[1]
        return {
            "images": torch.stack(imgs),
            "depths": torch.stack(depths),
            "intrinsics": torch.stack(Ks),
            "extrinsics": torch.stack(Es),
            "dynamic_mask": torch.zeros(S, H, W, dtype=torch.bool),
        }
