import glob
import os
from typing import List

import numpy as np
import torch

from .common import c2w_to_w2c, load_rgb, make_intrinsics, sliding_windows


def tartan_pose_to_c2w(xyzqxqyqzqw: np.ndarray) -> np.ndarray:
    # Follow MonST3R conversion exactly.
    z, x, y = xyzqxqyqzqw[:3]
    qz, qx, qy, qw = xyzqxqyqzqw[3:]
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float32,
    )
    c2w[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return c2w


class TartanAirClipDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        difficulty: str = "Hard",
        clip_len: int = 8,
        stride: int = 4,
    ):
        self.samples: List[dict] = []
        base = os.path.join(root, split)
        intrinsics = make_intrinsics(320.0, 320.0, 320.0, 240.0)

        seq_glob = os.path.join(base, "*", difficulty, "*")
        for seq in sorted(glob.glob(seq_glob)):
            rgb_dir = os.path.join(seq, "image_left")
            dpt_dir = os.path.join(seq, "depth_left")
            pose_file = os.path.join(seq, "pose_left.txt")
            if not (os.path.isdir(rgb_dir) and os.path.isdir(dpt_dir) and os.path.isfile(pose_file)):
                continue

            poses = np.loadtxt(pose_file, dtype=np.float32)
            rgbs = sorted(f for f in os.listdir(rgb_dir) if f.endswith("_left.png"))
            dpts = sorted(f for f in os.listdir(dpt_dir) if f.endswith("_left_depth.npy"))
            n = min(len(rgbs), len(dpts), poses.shape[0])
            for win in sliding_windows(n, clip_len, stride):
                self.samples.append(
                    {
                        "rgb_dir": rgb_dir,
                        "dpt_dir": dpt_dir,
                        "rgbs": rgbs,
                        "dpts": dpts,
                        "poses": poses,
                        "intrinsics": intrinsics,
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
            depth = np.load(os.path.join(s["dpt_dir"], s["dpts"][i])).astype(np.float32)
            depths.append(torch.from_numpy(depth))
            c2w = tartan_pose_to_c2w(s["poses"][i])
            Es.append(torch.from_numpy(c2w_to_w2c(c2w)))
            Ks.append(torch.from_numpy(s["intrinsics"]))

        S, H, W = len(imgs), depths[0].shape[0], depths[0].shape[1]
        return {
            "images": torch.stack(imgs),
            "depths": torch.stack(depths),
            "intrinsics": torch.stack(Ks),
            "extrinsics": torch.stack(Es),
            "dynamic_mask": torch.zeros(S, H, W, dtype=torch.bool),
        }
