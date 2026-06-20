import os
from typing import List

import h5py
import numpy as np
import torch

from .common import crop_resize_to_fixed, load_rgb_np, make_clip_aug_params, sample_random_window, sliding_windows

SPRING_BASELINE = 0.065


def read_dsp5(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if "disparity" not in f:
            raise IOError(f"Invalid dsp5 file (missing disparity): {path}")
        return f["disparity"][()]


class SpringClipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        clip_len: int = 8,
        stride: int = 4,
        expand_ratio: float = 0.0,
        max_depth: float = 100.0,
        target_hw: tuple = (294, 518),
        aug_crop: int = 0,
    ):
        self.clip_len = clip_len
        self.expand_ratio = expand_ratio
        self.max_depth = max_depth
        self.target_hw = target_hw
        self.aug_crop = aug_crop
        self.samples: List[dict] = []

        base = os.path.join(root, split)
        seqs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        for seq in seqs:
            seq_dir = os.path.join(base, seq)
            rgb_dir = os.path.join(seq_dir, "frame_left")
            dsp_dir = os.path.join(seq_dir, "disp1_left")
            ext_file = os.path.join(seq_dir, "cam_data", "extrinsics.txt")
            int_file = os.path.join(seq_dir, "cam_data", "intrinsics.txt")
            if not (os.path.isdir(rgb_dir) and os.path.isdir(dsp_dir) and os.path.isfile(ext_file) and os.path.isfile(int_file)):
                continue

            rgbs = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
            dsps = sorted(f for f in os.listdir(dsp_dir) if f.endswith(".dsp5"))
            ext = np.loadtxt(ext_file, dtype=np.float32).reshape(-1, 4, 4)
            intr = np.loadtxt(int_file, dtype=np.float32).reshape(-1, 4)
            n = min(len(rgbs), len(dsps), ext.shape[0], intr.shape[0])
            seq_entry = {
                "rgb_dir": rgb_dir,
                "dsp_dir": dsp_dir,
                "rgbs": rgbs,
                "dsps": dsps,
                "ext": ext,
                "intr": intr,
                "n": n,
            }
            if expand_ratio > 0:
                self.samples.append(seq_entry)
            else:
                for win in sliding_windows(n, clip_len, stride):
                    self.samples.append({**seq_entry, "idxs": win})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        idxs = sample_random_window(s["n"], self.clip_len, self.expand_ratio) \
            if self.expand_ratio > 0 else s["idxs"]
        imgs, depths, Ks, Es = [], [], [], []

        aug = make_clip_aug_params(self.aug_crop)
        for i in idxs:
            img = load_rgb_np(os.path.join(s["rgb_dir"], s["rgbs"][i]))

            fx, fy, cx, cy = [float(x) for x in s["intr"][i]]

            disp = read_dsp5(os.path.join(s["dsp_dir"], s["dsps"][i])).astype(np.float32)
            # Spring disparity maps are the same resolution as RGB; invalid pixels
            # (zero or near-zero disparity) map to infinite/negative depth — mark as 0.
            disp = np.where(disp <= 1e-8, np.nan, disp)
            depth = fx * SPRING_BASELINE / disp
            valid = np.isfinite(depth) & (depth > 0)
            depth = np.where(valid, np.clip(depth, 0, self.max_depth), 0.0).astype(np.float32)

            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

            img, depth, K = crop_resize_to_fixed(img, depth, K, self.target_hw, aug)

            imgs.append(torch.from_numpy(img.transpose(2, 0, 1).copy()))
            depths.append(torch.from_numpy(depth.copy()))
            Ks.append(torch.from_numpy(K))
            Es.append(torch.from_numpy(s["ext"][i][:3, :]))

        S, H, W = len(imgs), depths[0].shape[0], depths[0].shape[1]
        return {
            "images": torch.stack(imgs),
            "depths": torch.stack(depths),
            "intrinsics": torch.stack(Ks),
            "extrinsics": torch.stack(Es),
            "dynamic_mask": torch.zeros(S, H, W, dtype=torch.bool),
        }
