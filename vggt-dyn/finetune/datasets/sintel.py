import os
from typing import List

import numpy as np
import torch

from .common import (
    crop_resize_to_fixed,
    load_rgb_np,
    make_clip_aug_params,
    must_exist,
    sliding_windows,
)


TAG_FLOAT = 202021.25


def _read_sintel_depth(path: str) -> np.ndarray:
    """Read a Sintel .dpt depth map (see sintel-devkit's sintel_io.depth_read)."""
    with open(path, "rb") as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)[0]
        if abs(float(tag) - TAG_FLOAT) > 1e-4:
            raise ValueError(f"Wrong Sintel .dpt tag in {path}: {tag}")
        width = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        height = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        depth = np.fromfile(f, dtype=np.float32, count=width * height)
    return depth.reshape(height, width)


def _read_sintel_cam(path: str):
    """Read a Sintel .cam file: returns (K [3,3], extrinsics_w2c [3,4])."""
    with open(path, "rb") as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)[0]
        if abs(float(tag) - TAG_FLOAT) > 1e-4:
            raise ValueError(f"Wrong Sintel .cam tag in {path}: {tag}")
        intrinsics = np.fromfile(f, dtype=np.float64, count=9).reshape((3, 3))
        extrinsics = np.fromfile(f, dtype=np.float64, count=12).reshape((3, 4))
    return intrinsics.astype(np.float32), extrinsics.astype(np.float32)


class SintelClipDataset(torch.utils.data.Dataset):
    """MPI-Sintel clip dataset for validation.

    Reads `<root>/training/<pass>/<scene>/frame_XXXX.png` (RGB),
    `<root>/training/depth/<scene>/frame_XXXX.dpt` (GT depth) and
    `<root>/training/camdata_left/<scene>/frame_XXXX.cam` (GT intrinsics +
    world-to-cam extrinsics). Only the 16 scenes that ship GT depth are used.
    """

    def __init__(
        self,
        root: str,
        split: str = "final",  # Sintel image pass: "final" or "clean"
        clip_len: int = 8,
        stride: int = 4,
        max_depth: float = 80.0,
        target_hw: tuple = (294, 518),
        aug_crop: int = 0,
    ):
        base = os.path.join(root, "training")
        rgb_root = os.path.join(base, split)
        depth_root = os.path.join(base, "depth")
        cam_root = os.path.join(base, "camdata_left")
        if not os.path.isdir(rgb_root):
            raise FileNotFoundError(f"Sintel pass folder not found: {rgb_root}")
        if not os.path.isdir(depth_root):
            raise FileNotFoundError(f"Sintel depth folder not found: {depth_root}")

        self.max_depth = max_depth
        self.target_hw = target_hw
        self.aug_crop = aug_crop
        self.samples: List[dict] = []

        scenes = sorted(d for d in os.listdir(depth_root) if os.path.isdir(os.path.join(depth_root, d)))
        for scene in scenes:
            rgb_dir = os.path.join(rgb_root, scene)
            dpt_dir = os.path.join(depth_root, scene)
            cam_dir = os.path.join(cam_root, scene)
            must_exist(rgb_dir, "Missing Sintel RGB scene dir")
            must_exist(cam_dir, "Missing Sintel cam scene dir")

            rgbs = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
            dpts = sorted(f for f in os.listdir(dpt_dir) if f.endswith(".dpt"))
            cams = sorted(f for f in os.listdir(cam_dir) if f.endswith(".cam"))
            n = min(len(rgbs), len(dpts), len(cams))
            for win in sliding_windows(n, clip_len, stride):
                self.samples.append(
                    {
                        "rgb_dir": rgb_dir,
                        "dpt_dir": dpt_dir,
                        "cam_dir": cam_dir,
                        "rgbs": rgbs,
                        "dpts": dpts,
                        "cams": cams,
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

            depth = _read_sintel_depth(os.path.join(s["dpt_dir"], s["dpts"][i]))
            # Sky pixels have very large sentinel depth values; mark as invalid (0).
            valid = np.isfinite(depth) & (depth > 0) & (depth < 1e4)
            depth = np.where(valid, np.clip(depth, 0, self.max_depth), 0.0).astype(np.float32)

            K, extri = _read_sintel_cam(os.path.join(s["cam_dir"], s["cams"][i]))

            img, depth, K = crop_resize_to_fixed(img, depth, K, self.target_hw, aug)

            imgs.append(torch.from_numpy(img.transpose(2, 0, 1).copy()))
            depths.append(torch.from_numpy(depth.copy()))
            Ks.append(torch.from_numpy(K))
            Es.append(torch.from_numpy(extri))

        S, H, W = len(imgs), depths[0].shape[0], depths[0].shape[1]
        return {
            "images": torch.stack(imgs),
            "depths": torch.stack(depths),
            "intrinsics": torch.stack(Ks),
            "extrinsics": torch.stack(Es),
            "dynamic_mask": torch.zeros(S, H, W, dtype=torch.bool),
        }
