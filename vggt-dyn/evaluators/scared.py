"""evaluators/scared.py — SCARED benchmark evaluator.

Evaluation purpose: surgical scene depth accuracy (mm-level Chamfer).

Input (from run.py --output):
    pts3d.npy            [S, H, W, 3]
    dynamic_mask/        per-frame bool masks (optional)

GT:
    point_cloud.obj      dense GT mesh from SCARED dataset

Metrics:
    Chamfer @ 1 / 2 / 5 / 10 mm  (accuracy, completeness)
"""

import os
import numpy as np

from .base    import BaseEvaluator
from .metrics import chamfer_metrics, print_metrics


class SCaredEvaluator(BaseEvaluator):
    """Evaluates VGGT-Dyn outputs on the SCARED benchmark."""

    def run(self, args) -> dict:
        print("[eval:scared] loading VGGT-Dyn pts3d ...")
        pts = self.load_pts3d(args.output_dir)          # (N, 3)

        # ── remove dynamic pixels ────────────────────────────────────────────
        masks = self.load_dynamic_masks(args.output_dir)
        if masks is not None:
            static = ~masks.reshape(-1)
            pts    = pts[static]
            print(f"[eval:scared] kept {static.sum():,} / {len(static):,} static pixels")

        # ── load GT point cloud ──────────────────────────────────────────────
        print(f"[eval:scared] loading GT cloud from {args.gt_cloud} ...")
        gt_pts = _load_obj_cloud(args.gt_cloud)
        print(f"[eval:scared] GT: {len(gt_pts):,} pts  |  pred: {len(pts):,} pts")

        # ── optional median-scale alignment ──────────────────────────────────
        if args.align_scale_mode == "median":
            pred_c = np.median(pts,    axis=0)
            gt_c   = np.median(gt_pts, axis=0)
            pred_d = np.linalg.norm(pts    - pred_c, axis=1)
            gt_d   = np.linalg.norm(gt_pts - gt_c,   axis=1)
            scale  = np.median(gt_d) / (np.median(pred_d) + 1e-8)
            pts    = (pts - pred_c) * scale + gt_c
            print(f"[eval:scared] scale factor: {scale:.4f}")

        results = chamfer_metrics(pts, gt_pts)
        print_metrics(results, title="SCARED Results")

        path = self.save_results(results, args.output_dir, "scared")
        print(f"[eval:scared] saved → {path}")
        return results


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_obj_cloud(obj_path: str) -> np.ndarray:
    pts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v ") and "nan" not in line:
                parts = line.split()
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts, dtype=np.float32)
