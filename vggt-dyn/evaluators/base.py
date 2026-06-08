"""evaluators/base.py — BaseEvaluator abstract class.

All dataset-specific evaluators inherit from this.
"""

import os
import json
import numpy as np
from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Abstract base class for VGGT-Dyn evaluators.

    Sub-classes must implement:
        run(args) → dict  — execute evaluation, return metric dict

    Shared helpers provided here:
        load_depth_dir(output_dir)       → [S, H, W] float32
        load_pts3d(output_dir)           → [S*H*W, 3] float32
        load_dynamic_masks(output_dir)   → [S, H, W] bool | None
        save_results(results, path, tag)
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run(self, args) -> dict:
        """Run evaluation and return metric dict."""

    # ------------------------------------------------------------------
    # Shared I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_depth_dir(output_dir: str) -> np.ndarray:
        """Load per-frame depth npy files → [S, H, W] float32."""
        depth_dir = os.path.join(output_dir, "depth")
        if not os.path.isdir(depth_dir):
            raise FileNotFoundError(f"depth/ directory not found in {output_dir}")
        files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files in {depth_dir}")
        return np.stack(
            [np.load(os.path.join(depth_dir, f)) for f in files]
        ).astype(np.float32)   # [S, H, W]

    @staticmethod
    def load_pts3d(output_dir: str) -> np.ndarray:
        """Load pts3d.npy → (N, 3) flat array."""
        path = os.path.join(output_dir, "pts3d.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"pts3d.npy not found in {output_dir}")
        pts = np.load(path)   # [S, H, W, 3]
        return pts.reshape(-1, 3).astype(np.float32)

    @staticmethod
    def load_dynamic_masks(output_dir: str):
        """Load per-frame dynamic masks → bool [S, H, W] or None."""
        mask_dir = os.path.join(output_dir, "dynamic_mask")
        if not os.path.isdir(mask_dir):
            return None
        files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".npy"))
        if not files:
            return None
        return np.stack(
            [np.load(os.path.join(mask_dir, f)) for f in files]
        ).astype(bool)   # [S, H, W]

    @staticmethod
    def save_results(results: dict, output_dir: str, tag: str) -> str:
        """Save metric dict as JSON and return path."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"eval_{tag}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        return path
