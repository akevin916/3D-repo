"""
pose_utils.py — VGGT pose encoding utilities + delta pose operations.

Wraps vggt/vggt/utils/pose_enc.py and adds differentiable delta-pose
composition used during test-time optimization.
"""

import os
import sys

# Make the embedded VGGT repo importable as `vggt.*`
_VGGT_REPO = os.path.join(os.path.dirname(__file__), "../../vggt")
if _VGGT_REPO not in sys.path:
    sys.path.insert(0, _VGGT_REPO)

import torch
import roma

from vggt.utils.pose_enc import (
    pose_encoding_to_extri_intri,
    extri_intri_to_pose_encoding,
)


# ---------------------------------------------------------------------------
# Decode / encode
# ---------------------------------------------------------------------------

def decode_pose_enc(pose_enc: torch.Tensor, image_size_hw: tuple):
    """Decode VGGT pose encoding → R, T, K.

    Args:
        pose_enc : [S, 9] or [B, S, 9]
        image_size_hw : (H, W) in pixels

    Returns:
        R : [..., 3, 3]  rotation matrix (cam-from-world, OpenCV)
        T : [..., 3]     translation vector
        K : [..., 3, 3]  intrinsic matrix
    """
    squeeze = pose_enc.ndim == 2
    if squeeze:
        pose_enc = pose_enc.unsqueeze(0)  # [1, S, 9]

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw
    )
    # extrinsics: [B, S, 3, 4]
    R = extrinsics[..., :3, :3]   # [B, S, 3, 3]
    T = extrinsics[..., :3, 3]    # [B, S, 3]
    K = intrinsics                # [B, S, 3, 3]

    if squeeze:
        R = R.squeeze(0)  # [S, 3, 3]
        T = T.squeeze(0)  # [S, 3]
        K = K.squeeze(0)  # [S, 3, 3]

    return R, T, K


def encode_pose(R: torch.Tensor, T: torch.Tensor, K: torch.Tensor,
                image_size_hw: tuple) -> torch.Tensor:
    """Pack R, T, K back into VGGT pose_enc format [S, 9].

    Args:
        R : [S, 3, 3]
        T : [S, 3]
        K : [S, 3, 3]
        image_size_hw : (H, W)

    Returns:
        pose_enc : [S, 9]
    """
    extrinsics = torch.cat([R, T.unsqueeze(-1)], dim=-1)  # [S, 3, 4]
    # extri_intri_to_pose_encoding expects [B, S, ...]
    pose_enc = extri_intri_to_pose_encoding(
        extrinsics.unsqueeze(0),
        K.unsqueeze(0),
        image_size_hw,
    )
    return pose_enc.squeeze(0)  # [S, 9]


# ---------------------------------------------------------------------------
# Delta pose composition (for test-time optimization)
# ---------------------------------------------------------------------------

def compose_pose(
    R_base: torch.Tensor,
    T_base: torch.Tensor,
    delta_rotvec: torch.Tensor,
    delta_t: torch.Tensor,
):
    """Compose a base pose with a small residual delta pose.

    Applies the delta in the world frame:
        new_R = exp(delta_rotvec) @ R_base
        new_T = exp(delta_rotvec) @ T_base + delta_t

    Args:
        R_base      : [S, 3, 3]  base rotation (cam-from-world)
        T_base      : [S, 3]     base translation
        delta_rotvec: [S, 3]     rotation residual (axis-angle / Lie algebra)
        delta_t     : [S, 3]     translation residual

    Returns:
        R_new : [S, 3, 3]
        T_new : [S, 3]
    """
    delta_R = roma.rotvec_to_rotmat(delta_rotvec)         # [S, 3, 3]
    R_new = delta_R @ R_base                              # [S, 3, 3]
    T_new = (delta_R @ T_base.unsqueeze(-1)).squeeze(-1) + delta_t  # [S, 3]
    return R_new, T_new


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def RT_to_extrinsics(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Pack R, T → [..., 3, 4] extrinsics matrix."""
    return torch.cat([R, T.unsqueeze(-1)], dim=-1)


def K_inv(K: torch.Tensor) -> torch.Tensor:
    """Compute inverse intrinsics [..., 3, 3] → [..., 3, 3]."""
    return torch.linalg.inv(K)
