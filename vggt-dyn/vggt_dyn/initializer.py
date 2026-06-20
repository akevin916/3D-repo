"""
initializer.py — Convert VGGT output dict → optimizer initial state.

VGGT output shapes (B=1 for single-video inference):
    pose_enc         : [B, S, 9]
    depth            : [B, S, H, W, 1]
    depth_conf       : [B, S, H, W]
    world_points     : [B, S, H, W, 3]
    world_points_conf: [B, S, H, W]

Initializer output shapes (batch dim squeezed):
    R          : [S, 3, 3]    rotation (cam-from-world, OpenCV)
    T          : [S, 3]       translation
    K          : [S, 3, 3]    intrinsics
    depth      : [S, H, W]    depth in meters (always positive, VGGT uses exp activation)
    depth_conf : [S, H, W]    depth confidence
    anchor_pts : [S, H, W, 3] world-frame 3D points (frozen VGGT reference)
    anchor_conf: [S, H, W]    confidence weights for anchor loss
"""

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch

from vggt_dyn.utils.pose_utils import decode_pose_enc


class VGGTInitializer:
    """Parses a VGGT output to initialize VGGTDynOptimizer."""

    def __init__(self, vggt_output: dict, image_size_hw: tuple):
        """
        Args:
            vggt_output  : dict returned by ``VGGT.forward()``
            image_size_hw: (H, W) of the input images in pixels
        """
        self.image_size_hw = image_size_hw
        self._state = self._parse(vggt_output, image_size_hw)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(vggt_output: dict, image_size_hw: tuple) -> dict:
        pose_enc   = vggt_output["pose_enc"]            # [B, S, 9]
        depth      = vggt_output["depth"]               # [B, S, H, W, 1]
        depth_conf = vggt_output["depth_conf"]          # [B, S, H, W]
        world_pts  = vggt_output["world_points"]        # [B, S, H, W, 3]
        world_conf = vggt_output["world_points_conf"]   # [B, S, H, W]

        B = pose_enc.shape[0]
        assert B == 1, f"VGGTInitializer expects B=1, got B={B}"

        # squeeze batch dim
        pose_enc   = pose_enc.squeeze(0)          # [S, 9]
        depth      = depth.squeeze(0).squeeze(-1) # [S, H, W]  (remove trailing 1)
        depth_conf = depth_conf.squeeze(0)        # [S, H, W]
        world_pts  = world_pts.squeeze(0)         # [S, H, W, 3]
        world_conf = world_conf.squeeze(0)        # [S, H, W]

        R, T, K = decode_pose_enc(pose_enc, image_size_hw)  # [S,3,3], [S,3], [S,3,3]

        return {
            "R":           R,
            "T":           T,
            "K":           K,
            "depth":       depth,
            "depth_conf":  depth_conf,
            "anchor_pts":  world_pts,
            "anchor_conf": world_conf,
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def R(self) -> torch.Tensor:
        """[S, 3, 3] rotation matrices (cam-from-world)"""
        return self._state["R"]

    @property
    def T(self) -> torch.Tensor:
        """[S, 3] translation vectors"""
        return self._state["T"]

    @property
    def K(self) -> torch.Tensor:
        """[S, 3, 3] intrinsic matrices"""
        return self._state["K"]

    @property
    def depth(self) -> torch.Tensor:
        """[S, H, W] initial depth maps (always positive)"""
        return self._state["depth"]

    @property
    def depth_conf(self) -> torch.Tensor:
        """[S, H, W] depth confidence"""
        return self._state["depth_conf"]

    @property
    def anchor_pts(self) -> torch.Tensor:
        """[S, H, W, 3] frozen VGGT world points (anchor loss reference)"""
        return self._state["anchor_pts"]

    @property
    def anchor_conf(self) -> torch.Tensor:
        """[S, H, W] confidence weights for anchor loss"""
        return self._state["anchor_conf"]

    @property
    def S(self) -> int:
        return self._state["depth"].shape[0]

    @property
    def H(self) -> int:
        return self._state["depth"].shape[1]

    @property
    def W(self) -> int:
        return self._state["depth"].shape[2]

    def to(self, device):
        """Move all tensors to the specified device in-place."""
        self._state = {k: v.to(device) for k, v in self._state.items()}
        return self

    def as_dict(self) -> dict:
        """Return a shallow copy of the state dict."""
        return dict(self._state)

    @classmethod
    def from_dir(cls, output_dir: str, device=None) -> "VGGTInitializer":
        """Reconstruct initializer state from a previously saved run output directory.

        Expects the directory structure written by save_outputs():
            extrinsics.npy, intrinsics.npy, pts3d.npy,
            depth/XXXX.npy, depth_conf/XXXX.npy (optional), anchor_conf/XXXX.npy (optional)
        """
        extrinsics = np.load(os.path.join(output_dir, "extrinsics.npy"))  # [S, 3, 4]
        intrinsics  = np.load(os.path.join(output_dir, "intrinsics.npy")) # [S, 3, 3]
        pts3d       = np.load(os.path.join(output_dir, "pts3d.npy"))      # [S, H, W, 3]

        S = extrinsics.shape[0]
        H, W = pts3d.shape[1], pts3d.shape[2]

        depth_frames = [np.load(os.path.join(output_dir, "depth", f"{i:04d}.npy"))
                        for i in range(S)]
        depth = np.stack(depth_frames)  # [S, H, W]

        # anchor_conf: prefer saved anchor_conf/, fall back to depth_conf/, then ones
        anchor_conf_dir = os.path.join(output_dir, "anchor_conf")
        depth_conf_dir  = os.path.join(output_dir, "depth_conf")
        if os.path.isdir(anchor_conf_dir):
            frames = [np.load(os.path.join(anchor_conf_dir, f"{i:04d}.npy")) for i in range(S)]
            anchor_conf = np.stack(frames)
        elif os.path.isdir(depth_conf_dir):
            frames = [np.load(os.path.join(depth_conf_dir, f"{i:04d}.npy")) for i in range(S)]
            anchor_conf = np.stack(frames)
        else:
            anchor_conf = np.ones((S, H, W), dtype=np.float32)

        depth_conf_frames = [np.load(os.path.join(depth_conf_dir, f"{i:04d}.npy"))
                             for i in range(S)] if os.path.isdir(depth_conf_dir) else None
        depth_conf = np.stack(depth_conf_frames) if depth_conf_frames else anchor_conf

        def _t(arr):
            t = torch.from_numpy(arr.astype(np.float32))
            return t.to(device) if device is not None else t

        obj = cls.__new__(cls)
        obj.image_size_hw = (H, W)
        obj._state = {
            "R":           _t(extrinsics[:, :3, :3]),
            "T":           _t(extrinsics[:, :3, 3]),
            "K":           _t(intrinsics),
            "depth":       _t(depth),
            "depth_conf":  _t(depth_conf),
            "anchor_pts":  _t(pts3d),
            "anchor_conf": _t(anchor_conf),
        }
        return obj
