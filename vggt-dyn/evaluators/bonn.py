"""evaluators/bonn.py — Bonn RGB-D dataset evaluator.

Evaluation purpose: monocular depth estimation on dynamic indoor scenes.

Dataset structure (TUM RGB-D format):
    <scene>/
        rgb/          TIMESTAMP.png   — RGB frames
        depth/        TIMESTAMP.png   — GT depth, uint16, value/5000 = metres
        rgb.txt       timestamp  filename pairs
        depth.txt     timestamp  filename pairs
        groundtruth.txt  — GT camera trajectory (not used here)

Input (from run.py --output):
    depth/XXXX.npy   per-frame predicted depth [H, W]  (metres)

Metrics:
    AbsRel, SqRel, RMSE, RMSElog, δ1 / δ2 / δ3

Usage:
    python eval.py bonn \\
        --output_dir        outputs/bonn_balloon \\
        --scene_dir         data/bonn/rgbd_bonn_dataset/rgbd_bonn_balloon \\
        --align_scale_mode  median
"""

import os
import numpy as np

from .base    import BaseEvaluator
from .metrics import depth_metrics, print_metrics, scale_only_fit

BONN_DEPTH_SCALE = 5000.0   # uint16 value / 5000 = metres


class BonnEvaluator(BaseEvaluator):
    """Evaluates depth outputs on the Bonn RGB-D dynamic dataset."""

    def run(self, args) -> dict:
        pred_depths = self._load_pred_depths(args.output_dir)
        gt_depths   = self._load_gt_depths(args.scene_dir, len(pred_depths))

        if len(pred_depths) != len(gt_depths):
            raise ValueError(
                f"Frame count mismatch: pred={len(pred_depths)}, gt={len(gt_depths)}"
            )

        scene_name = os.path.basename(args.scene_dir.rstrip("/"))
        print(f"[eval:bonn] evaluating {len(pred_depths)} frames  "
              f"(scene: {scene_name}) ...")

        mode      = args.align_scale_mode
        max_depth = args.max_depth if args.max_depth > 0 else None

        if mode == "scale_and_shift":
            print("[eval:bonn] alignment: scale+shift (MonST3R protocol: align_with_lad2)")
        elif mode == "scale_only":
            print("[eval:bonn] alignment: per-sequence scale only (no shift)")
        elif mode == "single_frame":
            print("[eval:bonn] alignment: single-frame median scale (MonST3R single-frame protocol)")
        elif mode == "median":
            print("[eval:bonn] alignment: per-frame median scale")
        else:
            print("[eval:bonn] alignment: none")

        if max_depth is None:
            print("[eval:bonn] depth cap: none")
        else:
            print(f"[eval:bonn] depth cap: {max_depth} m")

        results = self._evaluate_sequence(
            pred_depths, gt_depths,
            align_median=(mode in ("median", "single_frame")),
            align_single_frame=(mode == "single_frame"),
            align_scale_only=(mode == "scale_only"),
            align_scale_shift=(mode == "scale_and_shift"),
            max_depth=max_depth,
            min_depth=args.min_depth,
        )
        if "d1" in results:
            results["delta_lt_1_25"] = float(results["d1"])
        if "d2" in results:
            results["delta_lt_1_25_sq"] = float(results["d2"])
        if "d3" in results:
            results["delta_lt_1_25_cu"] = float(results["d3"])

        title = f"Bonn Depth — VGGT-Dyn  [{scene_name}]"
        if mode == "scale_and_shift":
            title += " [scale+shift / MonST3R protocol]"
        elif mode == "scale_only":
            title += " [scale-only / per-sequence]"
        elif mode == "single_frame":
            title += " [single-frame / MonST3R protocol]"
        elif mode == "median":
            title += " [median / per-frame]"
        print_metrics(results, title=title)

        if mode == "scale_and_shift":
            suffix = f"bonn_{scene_name}_scale_shift"
        elif mode == "scale_only":
            suffix = f"bonn_{scene_name}_scale_only"
        elif mode == "single_frame":
            suffix = f"bonn_{scene_name}_single_frame"
        elif mode == "median":
            suffix = f"bonn_{scene_name}_median"
        else:
            suffix = f"bonn_{scene_name}"
        path = self.save_results(results, args.output_dir, suffix)
        print(f"[eval:bonn] saved → {path}")
        return results

    # ------------------------------------------------------------------
    # Core evaluation loop
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_sequence(
        pred_list,
        gt_list,
        align_median: bool,
        align_single_frame: bool,
        align_scale_only: bool,
        align_scale_shift: bool,
        max_depth=10.0,
        min_depth=0.1,
    ) -> dict:
        def _valid_mask(gt, pred=None):
            v = gt > min_depth
            if max_depth is not None:
                v &= gt < max_depth
            if pred is not None:
                v &= pred > 0
            return v

        pred_list_r = []
        for pred, gt in zip(pred_list, gt_list):
            if pred.shape != gt.shape:
                pred = _resize_depth(pred, gt.shape)
            pred_list_r.append(pred)

        if align_scale_shift:
            all_pred = np.concatenate([p.ravel() for p in pred_list_r])
            all_gt = np.concatenate([g.ravel() for g in gt_list])
            all_valid = np.concatenate([
                _valid_mask(g, p).ravel() for p, g in zip(pred_list_r, gt_list)
            ])
            p_valid = all_pred[all_valid]
            g_valid = all_gt[all_valid]
            s_val, t_val = _scale_shift_fit(p_valid, g_valid)
            print(f"[eval:bonn] MonST3R scale+shift: s={s_val:.4f}, t={t_val:.4f}")

            accum = {k: [] for k in ("abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3")}
            weights = []
            for pred, gt in zip(pred_list_r, gt_list):
                valid = _valid_mask(gt, pred)
                if valid.sum() == 0:
                    continue
                pred_v = pred[valid].astype(np.float64) * s_val + t_val
                gt_v = gt[valid].astype(np.float64)
                pred_v = np.clip(pred_v, 1e-5, max_depth if max_depth else 1e9)
                m = depth_metrics(pred_v, gt_v)
                n = int(valid.sum())
                for k in accum:
                    accum[k].append(m[k])
                weights.append(n)
            if not accum["abs_rel"]:
                raise RuntimeError("No valid frames evaluated")
            w = np.array(weights, dtype=np.float64)
            out = {k: float(np.average(accum[k], weights=w)) for k in accum}
            out["valid_pixels"] = int(np.sum(weights))
            return out

        if align_scale_only:
            all_pred = np.concatenate([p.ravel() for p in pred_list_r])
            all_gt = np.concatenate([g.ravel() for g in gt_list])
            all_valid = np.concatenate([
                _valid_mask(g, p).ravel() for p, g in zip(pred_list_r, gt_list)
            ])
            p_valid = all_pred[all_valid]
            g_valid = all_gt[all_valid]
            s_val = scale_only_fit(p_valid, g_valid)
            print(f"[eval:bonn] scale-only factor: s={s_val:.4f}")

            accum = {k: [] for k in ("abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3")}
            weights = []
            for pred, gt in zip(pred_list_r, gt_list):
                valid = _valid_mask(gt, pred)
                if valid.sum() == 0:
                    continue
                pred_v = pred[valid].astype(np.float64) * s_val
                gt_v = gt[valid].astype(np.float64)
                pred_v = np.clip(pred_v, 1e-5, max_depth if max_depth else 1e9)
                m = depth_metrics(pred_v, gt_v)
                n = int(valid.sum())
                for k in accum:
                    accum[k].append(m[k])
                weights.append(n)
            if not accum["abs_rel"]:
                raise RuntimeError("No valid frames evaluated")
            w = np.array(weights, dtype=np.float64)
            out = {k: float(np.average(accum[k], weights=w)) for k in accum}
            out["valid_pixels"] = int(np.sum(weights))
            return out

        accum = {k: [] for k in ("abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3")}
        weights = []
        for pred, gt in zip(pred_list_r, gt_list):
            valid = _valid_mask(gt, pred)
            if valid.sum() == 0:
                continue

            pred_v = pred[valid].astype(np.float64)
            gt_v = gt[valid].astype(np.float64)
            pred_v = np.clip(pred_v, 1e-4, max_depth if max_depth else 1e9)

            if align_median:
                scale = np.median(gt_v) / (np.median(pred_v) + 1e-8)
                pred_v = pred_v * scale

            pred_v = np.clip(pred_v, 1e-4, max_depth if max_depth else 1e9)
            m = depth_metrics(pred_v, gt_v)
            for k in accum:
                accum[k].append(m[k])
            weights.append(int(valid.sum()))

        if not accum["abs_rel"]:
            raise RuntimeError("No valid frames evaluated")

        if align_single_frame:
            w = np.array(weights, dtype=np.float64)
            out = {k: float(np.average(accum[k], weights=w)) for k in accum}
        else:
            out = {k: float(np.mean(v)) for k, v in accum.items()}
        out["valid_pixels"] = int(np.sum(weights))
        return out

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_pred_depths(self, output_dir: str) -> list:
        depth_dir = os.path.join(output_dir, "depth")
        if not os.path.isdir(depth_dir):
            depth_dir = output_dir
        files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".npy"))
        if not files:
            raise FileNotFoundError(f"No depth .npy files found in {depth_dir}")
        return [np.load(os.path.join(depth_dir, f)).astype(np.float32)
                for f in files]

    @staticmethod
    def _load_gt_depths(scene_dir: str, n_frames: int) -> list:
        """Load GT depth from Bonn scene directory.

        MonST3R depth protocol on Bonn commonly uses rgb_110/depth_110. To
        stay consistent, this loader prefers depth_110 when present, then
        falls back to depth. If depth*.txt exists, timestamp order is used.
        """
        candidates = [
            ("depth_110", "depth_110.txt"),
            ("depth", "depth.txt"),
        ]

        files = []
        for depth_subdir, depth_txt_name in candidates:
            depth_dir = os.path.join(scene_dir, depth_subdir)
            depth_txt = os.path.join(scene_dir, depth_txt_name)
            if not os.path.isdir(depth_dir):
                continue

            if os.path.isfile(depth_txt):
                cand = _parse_tum_txt(depth_txt, scene_dir)
            else:
                cand = [
                    os.path.join(depth_dir, f)
                    for f in sorted(os.listdir(depth_dir))
                    if f.endswith(".png")
                ]

            if not cand:
                continue

            files = cand
            print(f"[eval:bonn] GT source: {depth_subdir}/")
            break

        if not files:
            raise FileNotFoundError(
                f"No GT depth files found under {scene_dir}/depth_110 or {scene_dir}/depth"
            )

        if len(files) != n_frames:
            print(
                f"[eval:bonn] WARNING: GT has {len(files)} frames, pred has "
                f"{n_frames}.  Using first {min(len(files), n_frames)}."
            )
            files = files[: min(len(files), n_frames)]

        return [_load_bonn_depth_png(p) for p in files]

    # ------------------------------------------------------------------
    # Side-by-side comparison  (reuse from KITTIEvaluator)
    # ------------------------------------------------------------------

    @staticmethod
    def _print_comparison(a: dict, b: dict, name_a: str, name_b: str):
        keys = list(a.keys())
        w    = max(len(k) for k in keys) + 2
        nw   = max(len(name_a), len(name_b)) + 2
        print(f"\n{'─' * (w + nw * 2 + 10)}")
        print(f"  {'Metric':<{w}}  {name_a:>{nw}}  {name_b:>{nw}}  {'Δ':>8}")
        print(f"{'─' * (w + nw * 2 + 10)}")
        for k in keys:
            va, vb = a[k], b[k]
            delta  = va - vb
            sign   = "+" if delta > 0 else ""
            print(f"  {k:<{w}}  {va:>{nw}.4f}  {vb:>{nw}.4f}  {sign}{delta:>7.4f}")
        print(f"{'─' * (w + nw * 2 + 10)}\n")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_tum_txt(txt_path: str, scene_dir: str) -> list:
    """Parse TUM-style txt file → list of absolute file paths (sorted by timestamp)."""
    entries = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                entries.append((float(parts[0]),
                                 os.path.join(scene_dir, parts[1])))
    entries.sort(key=lambda x: x[0])
    return [p for _, p in entries]


def _load_bonn_depth_png(path: str) -> np.ndarray:
    """Load Bonn uint16 depth PNG → float32 metres (0 = invalid)."""
    try:
        from PIL import Image
        img = np.array(Image.open(path), dtype=np.float32)
    except ImportError:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    depth = img / BONN_DEPTH_SCALE
    depth[depth <= 0] = 0.0
    return depth


def _resize_depth(depth: np.ndarray, target_shape) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(depth, (target_shape[1], target_shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        return np.array(
            Image.fromarray(depth).resize(
                (target_shape[1], target_shape[0]), Image.BILINEAR
            ), dtype=np.float32
        )


def _scale_shift_fit(pred_valid: np.ndarray, gt_valid: np.ndarray):
    import torch

    p_t = torch.tensor(pred_valid, dtype=torch.float32)
    g_t = torch.tensor(gt_valid, dtype=torch.float32)
    s_init = float(np.median(gt_valid) / (np.median(pred_valid) + 1e-8))
    s_p = torch.tensor([s_init], requires_grad=True, dtype=torch.float32)
    t_p = torch.tensor([0.0], requires_grad=True, dtype=torch.float32)
    opt = torch.optim.Adam([s_p, t_p], lr=1e-4)

    prev = None
    for _ in range(1000):
        opt.zero_grad()
        loss = torch.abs(s_p * p_t + t_p - g_t).sum()
        loss.backward()
        opt.step()
        cur = loss.item()
        if prev is not None and abs(prev - cur) < 1e-6:
            break
        prev = cur

    return float(s_p.detach()), float(t_p.detach())
