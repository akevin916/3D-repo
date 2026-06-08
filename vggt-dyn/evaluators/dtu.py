"""evaluators/dtu.py — DTU benchmark evaluator.

Evaluation purpose: multi-view 3D reconstruction accuracy (Chamfer distance).

Input (from run.py --output):
    pts3d.npy            [S, H, W, 3]
    extrinsics.npy       [S, 3, 4]
    dynamic_mask/        (optional)

GT alignment:
    Umeyama Sim(3) using predicted vs GT camera centres from pos_XXX.txt

Metrics:
    acc_mm, comp_mm, overall_mm  (via DTUeval-python)
"""

import os
import sys
import json
import subprocess
import numpy as np
import glob as _glob

from .base import BaseEvaluator


class DTUEvaluator(BaseEvaluator):
    """Evaluates VGGT-Dyn outputs on the DTU benchmark."""

    def run(self, args) -> dict:
        pts = self.load_pts3d(args.output_dir)          # (N, 3)

        # ── remove dynamic pixels ────────────────────────────────────────────
        masks = self.load_dynamic_masks(args.output_dir)
        if masks is not None:
            static = ~masks.reshape(-1)
            pts    = pts[static]
            print(f"[eval:dtu] kept {static.sum():,} / {len(static):,} static pixels")

        # ── Umeyama Sim(3) alignment ─────────────────────────────────────────
        ext_path = os.path.join(args.output_dir, "extrinsics.npy")
        if args.align_scale_mode == "sim3" and args.calib_dir and args.images and os.path.isfile(ext_path):
            extrinsics   = np.load(ext_path)             # [S, 3, 4]
            R_pred       = extrinsics[:, :3, :3]
            t_pred       = extrinsics[:, :3,  3]
            centers_pred = np.array([-R_pred[i].T @ t_pred[i]
                                     for i in range(len(R_pred))])

            image_paths = sorted(_glob.glob(args.images))
            cam_indices = [int(os.path.basename(p).split("_")[1])
                           for p in image_paths]

            centers_gt_list, valid_idx = [], []
            for i, cam_idx in enumerate(cam_indices):
                pos_file = os.path.join(args.calib_dir, f"pos_{cam_idx:03d}.txt")
                if os.path.isfile(pos_file):
                    _, R_gt, t_gt = _decompose_proj(pos_file)
                    centers_gt_list.append(-R_gt.T @ t_gt)
                    valid_idx.append(i)

            if len(centers_gt_list) >= 2:
                centers_gt    = np.array(centers_gt_list)
                centers_pred_v = centers_pred[valid_idx]
                s, R_a, t_a  = _umeyama(centers_pred_v, centers_gt)
                print(f"[eval:dtu] Umeyama Sim(3)  s = {s:.4f}")
                pts = (s * (pts @ R_a.T) + t_a).astype(np.float32)
            else:
                print("[eval:dtu] WARNING: not enough GT cameras — skipping Sim(3)")
        else:
            print("[eval:dtu] WARNING: --align_scale_mode sim3 or --calib_dir / --images not provided — no alignment.")

        # ── save PLY ─────────────────────────────────────────────────────────
        ply_path = os.path.join(args.output_dir, f"pred_scan{args.scan_id:03d}.ply")
        _save_ply(pts, ply_path)
        print(f"[eval:dtu] PLY saved → {ply_path}  ({len(pts):,} pts)")

        # ── DTUeval-python ────────────────────────────────────────────────────
        script_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dtu_eval_script = os.path.join(script_dir, "..", "DTUeval-python", "eval.py")
        if not os.path.isfile(dtu_eval_script):
            print(f"[eval:dtu] DTUeval-python not found at {dtu_eval_script}")
            return {}

        dataset_dir = args.dataset_dir or os.path.join(
            script_dir, "..", "data", "SampleSet", "MVS Data"
        )
        vis_dir = os.path.join(args.output_dir, f"dtu_vis_scan{args.scan_id}")
        os.makedirs(vis_dir, exist_ok=True)

        cmd = [
            sys.executable, dtu_eval_script,
            "--data",        ply_path,
            "--scan",        str(args.scan_id),
            "--mode",        "pcd",
            "--dataset_dir", dataset_dir,
            "--vis_out_dir", vis_dir,
        ]
        print("[eval:dtu] running DTUeval-python ...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print("[eval:dtu] error:", result.stderr[-300:])
            return {}

        lines = result.stdout.strip().split("\n")
        try:
            nums    = [float(x) for x in lines[-1].split()]
            metrics = {"acc_mm": nums[0], "comp_mm": nums[1], "overall_mm": nums[2]}
        except Exception:
            print("[eval:dtu] could not parse DTUeval output")
            return {}

        from .metrics import print_metrics
        print_metrics(metrics, title=f"DTU scan{args.scan_id} Results")

        path = self.save_results(metrics, args.output_dir, f"dtu_scan{args.scan_id}")
        print(f"[eval:dtu] saved → {path}")
        return metrics


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _rq_decomp(M: np.ndarray):
    n = M.shape[0]
    P = np.fliplr(np.eye(n))
    Qt, Rt = np.linalg.qr((P @ M @ P).T)
    return P @ Rt.T @ P, P @ Qt.T @ P


def _decompose_proj(proj_file: str):
    with open(proj_file) as f:
        lines = [l.strip() for l in f if l.strip()]
    P = np.array([[float(x) for x in l.split()] for l in lines], dtype=np.float64)
    assert P.shape == (3, 4)
    K, Rot = _rq_decomp(P[:, :3])
    signs  = np.diag(np.sign(np.diag(K)))
    K, Rot = K @ signs, signs @ Rot
    K     /= K[2, 2]
    K[0, 1] = 0.0
    t = np.linalg.inv(K) @ P[:, 3]
    if np.linalg.det(Rot) < 0:
        Rot, t = -Rot, -t
    return K.astype(np.float32), Rot.astype(np.float32), t.astype(np.float32)


def _umeyama(src: np.ndarray, dst: np.ndarray):
    N    = src.shape[0]
    mu_s = src.mean(0);  mu_d = dst.mean(0)
    sc   = src - mu_s;   dc   = dst - mu_d
    var_s = np.mean(np.sum(sc ** 2, axis=1))
    H     = (sc.T @ dc) / N
    U, D, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        S[2, 2] = -1
    R = Vt.T @ S @ U.T
    s = np.sum(D * np.diag(S)) / max(var_s, 1e-12)
    t = mu_d - s * R @ mu_s
    return float(s), R, t


def _save_ply(pts: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
