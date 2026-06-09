"""datasets.py — PyTorch Datasets for VGGT dynamic fine-tuning.

Provides sliding-window clip datasets for:
  - MPI-Sintel (depth .dpt, poses .cam, optional GT flow .flo)
  - Bonn RGB-D  (depth PNG uint16, poses TUM groundtruth.txt)

Extracted from finetune_dynamic.py for modularity.
"""

import os
import struct
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# ── Sintel constants ────────────────────────────────────────────────────────
SINTEL_FX  = 1120.0
SINTEL_FY  = 1120.0
SINTEL_CX  = 511.5
SINTEL_CY  = 217.5
SINTEL_H   = 436
SINTEL_W   = 1024

SINTEL_DPT_TAG = 202021.25
SINTEL_FLO_TAG = 202021.25


def _read_sintel_dpt(path: str) -> np.ndarray:
    """Read a Sintel .dpt depth file → float32 [H, W] in metres."""
    with open(path, "rb") as f:
        tag = struct.unpack("<f", f.read(4))[0]
        if abs(tag - SINTEL_DPT_TAG) > 1e-4:
            raise ValueError(f"Bad Sintel dpt tag {tag} in {path}")
        W, H = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(W * H * 4), dtype=np.float32).copy()
    return data.reshape(H, W)


def _read_sintel_flo(path: str) -> Optional[np.ndarray]:
    """Read a Sintel .flo optical flow → float32 [H, W, 2] (u, v)."""
    with open(path, "rb") as f:
        tag = struct.unpack("<f", f.read(4))[0]
        if abs(tag - SINTEL_FLO_TAG) > 1e-4:
            return None
        W, H = struct.unpack("<II", f.read(8))
        data = np.frombuffer(f.read(W * H * 8), dtype=np.float32).copy()
    return data.reshape(H, W, 2)


def _read_sintel_cam(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read a Sintel .cam file → (intrinsics [3,3], extrinsics [3,4]) float32."""
    with open(path, "rb") as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)[0]
        if abs(float(tag) - SINTEL_DPT_TAG) > 1e-4:
            raise ValueError(f"Bad Sintel cam tag in {path}")
        K = np.fromfile(f, dtype=np.float64, count=9 ).reshape(3, 3)
        E = np.fromfile(f, dtype=np.float64, count=12).reshape(3, 4)
    return K.astype(np.float32), E.astype(np.float32)


class SintelClipDataset(torch.utils.data.Dataset):
    """Sliding-window clips of *clip_len* frames from Sintel sequences.

    Each item is a dict with:
      images       [S, 3, H, W]   float32, [0, 1]
      depths       [S, H, W]      float32, metres
      extrinsics   [S, 3, 4]      world-to-camera
      intrinsics   [S, 3, 3]
      dynamic_mask [S, H, W]      bool, True = dynamic pixel
    """

    def __init__(
        self,
        image_dir: str,
        depth_dir: str,
        cam_dir:   str,
        flow_dir:  Optional[str] = None,
        sequences: Optional[List[str]] = None,
        clip_len:  int   = 8,
        stride:    int   = 4,
        dyn_thresh: float = 1.5,
        img_hw:    Tuple[int, int] = (SINTEL_H, SINTEL_W),
    ):
        self.clip_len   = clip_len
        self.stride     = stride
        self.dyn_thresh = dyn_thresh
        self.img_hw     = img_hw

        if sequences is None:
            sequences = sorted(
                d for d in os.listdir(image_dir)
                if os.path.isdir(os.path.join(image_dir, d))
            )

        self.clips: List[Tuple[str, List[int]]] = []
        for seq in sequences:
            frames = sorted(
                f for f in os.listdir(os.path.join(image_dir, seq))
                if f.endswith(".png")
            )
            n = len(frames)
            for start in range(0, max(1, n - clip_len + 1), stride):
                end = min(start + clip_len, n)
                if end - start < 2:
                    continue
                self.clips.append((seq, list(range(start, end))))

        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.cam_dir   = cam_dir
        self.flow_dir  = flow_dir

        log.info(f"[SintelClipDataset] {len(sequences)} seqs → {len(self.clips)} clips")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        from PIL import Image as _PIL
        seq, frame_ids = self.clips[idx]
        S = len(frame_ids)

        imgs, depths, Ks, Es, dyn_masks = [], [], [], [], []

        all_frames = sorted(
            f for f in os.listdir(os.path.join(self.image_dir, seq))
            if f.endswith(".png")
        )
        all_dpts = sorted(
            f for f in os.listdir(os.path.join(self.depth_dir, seq))
            if f.endswith(".dpt")
        )
        all_cams = sorted(
            f for f in os.listdir(os.path.join(self.cam_dir, seq))
            if f.endswith(".cam")
        )
        has_flow = (
            self.flow_dir is not None
            and os.path.isdir(os.path.join(self.flow_dir, seq))
        )
        all_flos = []
        if has_flow:
            all_flos = sorted(
                f for f in os.listdir(os.path.join(self.flow_dir, seq))
                if f.endswith(".flo")
            )

        for fi in frame_ids:
            img_path = os.path.join(self.image_dir, seq, all_frames[fi])
            img_np = np.array(_PIL.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(img_np).permute(2, 0, 1))

            dpt_path = os.path.join(self.depth_dir, seq, all_dpts[fi])
            depths.append(torch.from_numpy(_read_sintel_dpt(dpt_path)))

            cam_path = os.path.join(self.cam_dir, seq, all_cams[fi])
            K_np, E_np = _read_sintel_cam(cam_path)
            Ks.append(torch.from_numpy(K_np))
            Es.append(torch.from_numpy(E_np))

            if has_flow and fi < len(all_flos):
                flo_path = os.path.join(self.flow_dir, seq, all_flos[fi])
                flo = _read_sintel_flo(flo_path)
                if flo is not None:
                    mag = np.sqrt((flo ** 2).sum(axis=-1))
                    med = float(np.median(mag[mag > 0])) if (mag > 0).any() else 1.0
                    dyn = torch.from_numpy(mag > self.dyn_thresh * med)
                else:
                    dyn = torch.zeros(self.img_hw, dtype=torch.bool)
            else:
                dyn = torch.zeros(self.img_hw, dtype=torch.bool)
            dyn_masks.append(dyn)

        return {
            "images":       torch.stack(imgs),
            "depths":       torch.stack(depths),
            "extrinsics":   torch.stack(Es),
            "intrinsics":   torch.stack(Ks),
            "dynamic_mask": torch.stack(dyn_masks),
            "seq":          seq,
        }


# ── Bonn RGB-D dataset ──────────────────────────────────────────────────────

BONN_DEPTH_SCALE = 5000.0   # uint16 / 5000 = metres
BONN_FX, BONN_FY = 542.822642, 542.576870
BONN_CX, BONN_CY = 315.593520, 237.756098


def _read_tum_trajectory(path: str):
    """Parse TUM format groundtruth.txt → dict {timestamp_str: [tx ty tz qx qy qz qw]}."""
    poses = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            poses[parts[0]] = [float(x) for x in parts[1:8]]
    return poses


def _tum_pose_to_extrinsics(pose7: list) -> np.ndarray:
    """Convert TUM [tx ty tz qx qy qz qw] (cam-to-world) → 3×4 world-to-cam."""
    from scipy.spatial.transform import Rotation as ScipyR
    tx, ty, tz, qx, qy, qz, qw = pose7
    R_c2w = ScipyR.from_quat([qx, qy, qz, qw]).as_matrix()
    t_c2w = np.array([tx, ty, tz])
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R_w2c
    E[:3,  3] = t_w2c
    return E[:3, :]


class BonnClipDataset(torch.utils.data.Dataset):
    """Bonn RGB-D sliding-window clips.

    Each item:
      images       [S, 3, H, W]
      depths       [S, H, W]
      extrinsics   [S, 3, 4]
      intrinsics   [S, 3, 3]
      dynamic_mask [S, H, W]   (all-False; Bonn has no GT optical flow)
    """

    def __init__(
        self,
        scene_dir: str,
        clip_len:  int  = 8,
        stride:    int  = 4,
        max_depth: float = 10.0,
        img_hw:    Tuple[int, int] = (480, 640),
    ):
        self.scene_dir = scene_dir
        self.clip_len  = clip_len
        self.stride    = stride
        self.max_depth = max_depth
        self.img_hw    = img_hw

        rgb_list   = self._read_txt(os.path.join(scene_dir, "rgb.txt"))
        depth_list = self._read_txt(os.path.join(scene_dir, "depth.txt"))
        gt_poses   = _read_tum_trajectory(os.path.join(scene_dir, "groundtruth.txt"))

        self.frames = []
        for ts_rgb, rgb_rel in rgb_list:
            best_gt  = min(gt_poses.keys(), key=lambda t: abs(float(t) - float(ts_rgb)))
            best_dpt = min(depth_list, key=lambda x: abs(float(x[0]) - float(ts_rgb)))[1]
            if abs(float(best_gt) - float(ts_rgb)) > 0.05:
                continue
            E = _tum_pose_to_extrinsics(gt_poses[best_gt])
            self.frames.append((
                os.path.join(scene_dir, rgb_rel),
                os.path.join(scene_dir, best_dpt),
                E,
            ))

        self.clips = []
        n = len(self.frames)
        for start in range(0, max(1, n - clip_len + 1), stride):
            end = min(start + clip_len, n)
            if end - start < 2:
                continue
            self.clips.append(list(range(start, end)))

        log.info(f"[BonnClipDataset] {len(self.frames)} frames → {len(self.clips)} clips")

    @staticmethod
    def _read_txt(path: str):
        pairs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        from PIL import Image as _PIL
        frame_ids = self.clips[idx]
        imgs, depths, Es = [], [], []
        H, W = self.img_hw
        fx, fy = BONN_FX, BONN_FY
        cx, cy = BONN_CX, BONN_CY
        S = len(frame_ids)

        for fi in frame_ids:
            rgb_path, dpt_path, E = self.frames[fi]

            img_np = np.array(_PIL.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(img_np).permute(2, 0, 1))

            dpt_np = np.array(_PIL.open(dpt_path), dtype=np.uint16).astype(np.float32)
            dpt_np = dpt_np / BONN_DEPTH_SCALE
            dpt_np = np.clip(dpt_np, 0, self.max_depth)
            depths.append(torch.from_numpy(dpt_np))

            Es.append(torch.from_numpy(E))

        K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        K_tensor = torch.from_numpy(K_np).unsqueeze(0).expand(S, -1, -1)

        return {
            "images":       torch.stack(imgs),
            "depths":       torch.stack(depths),
            "extrinsics":   torch.stack(Es),
            "intrinsics":   K_tensor.clone(),
            "dynamic_mask": torch.zeros(S, H, W, dtype=torch.bool),
        }
