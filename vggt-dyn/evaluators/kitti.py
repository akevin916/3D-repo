"""evaluators/kitti.py — KITTI depth evaluator (Eigen split).

Evaluation purpose: monocular depth estimation accuracy on KITTI.

Input (from run.py --output):
    depth/XXXX.npy       per-frame refined depth [H, W]  (metric depth, metres)

GT:
    KITTI projected LiDAR depth  (uint16 PNG, value / 256 = depth in metres)
    OR depth from the "depth_selection" val set.

Protocol (Eigen / Garg, standard in MonST3R, Monodepth2, ...):
    - Depth cap: 0 < d < 80 m  (set --max_depth)
    - Eigen crop on evaluation mask
    - Median per-frame scale alignment  (set --align_scale)

Metrics:
    AbsRel, SqRel, RMSE, RMSElog, δ1, δ2, δ3

Usage:
    python eval.py kitti \\
        --output_dir        outputs/my_kitti_run \\
        --gt_dir            data/KITTI/depth_gt \\
        --align_scale_mode  single_frame
"""

import os
import numpy as np

from .base    import BaseEvaluator
from .metrics import depth_metrics, print_metrics, scale_only_fit


class KITTIEvaluator(BaseEvaluator):
    """Evaluates depth outputs on the KITTI Eigen split.

    GT depth directory layout (two common formats are supported):
        Format A — flat directory, one PNG per frame:
            <gt_dir>/XXXXXXXX.png      (uint16, value / 256 = metres)
        Format B — KITTI raw structure, GT from velodyne:
            <gt_dir>/<date>/<drive>/proj_depth/groundtruth/image_02/XXXXXXXXXX.png
        Format C — pre-computed .npy files:
            <gt_dir>/XXXX.npy          (float32, metres)

    Predicted depth directory (from run.py):
        <output_dir>/depth/XXXX.npy   (float32, metres)
    """

    # Eigen crop constants (ratios, resolution-independent)
    _CROP_T = 0.40810811
    _CROP_B = 0.99189189
    _CROP_L = 0.03594771
    _CROP_R = 0.96405229

    def run(self, args) -> dict:
        pred_depths = self._load_pred_depths(args.output_dir)
        gt_depths   = self._load_gt_depths(
            args.gt_dir, len(pred_depths),
            drive=getattr(args, "drive", None),
        )

        if len(pred_depths) != len(gt_depths):
            raise ValueError(
                f"Frame count mismatch: pred={len(pred_depths)}, gt={len(gt_depths)}"
            )

        print(f"[eval:kitti] evaluating {len(pred_depths)} frames ...")

        mode      = args.align_scale_mode
        max_depth = args.max_depth if args.max_depth > 0 else None

        if mode == "scale_and_shift":
            print("[eval:kitti] alignment: scale+shift (MonST3R protocol: align_with_lad2)")
        elif mode == "scale_only":
            print("[eval:kitti] alignment: per-sequence scale only (no shift)")
        elif mode == "single_frame":
            print("[eval:kitti] alignment: single-frame median scale (MonST3R single-frame protocol)")
        elif mode == "median":
            print("[eval:kitti] alignment: per-frame median scale")
        else:
            print("[eval:kitti] alignment: none")

        if max_depth is None:
            print("[eval:kitti] depth cap: none (MonST3R protocol)")
        else:
            print(f"[eval:kitti] depth cap: {max_depth} m")

        results = self._evaluate_sequence(
            pred_depths, gt_depths,
            align_median=(mode in ("median", "single_frame")),
            align_single_frame=(mode == "single_frame"),
            align_scale_only=(mode == "scale_only"),
            align_scale_shift=(mode == "scale_and_shift"),
            max_depth=max_depth,
            use_eigen_crop=not args.no_eigen_crop,
        )
        # Explicit aliases for readability in saved JSON/reporting.
        if "d1" in results:
            results["delta_lt_1_25"] = float(results["d1"])
        if "d2" in results:
            results["delta_lt_1_25_sq"] = float(results["d2"])
        if "d3" in results:
            results["delta_lt_1_25_cu"] = float(results["d3"])
        title = "KITTI Depth — VGGT-Dyn"
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
            suffix = "kitti_scale_shift"
        elif mode == "scale_only":
            suffix = "kitti_scale_only"
        elif mode == "single_frame":
            suffix = "kitti_single_frame"
        elif mode == "median":
            suffix = "kitti_median"
        else:
            suffix = "kitti"
        path = self.save_results(results, args.output_dir, suffix)
        print(f"[eval:kitti] saved → {path}")
        return results

    # ------------------------------------------------------------------
    # Core evaluation loop
    # ------------------------------------------------------------------

    def _evaluate_sequence(
        self,
        pred_list: list,
        gt_list: list,
        align_median: bool,
        align_single_frame: bool,
        align_scale_only: bool,
        max_depth,            # float or None
        use_eigen_crop: bool,
        align_scale_shift: bool = False,
    ) -> dict:
        """Compute per-frame metrics and aggregate.

        If align_scale_shift=True, perform a single per-SEQUENCE scale+shift
        alignment (MonST3R protocol: all frames stacked → one global s, t via
        Adam L1 minimisation), then evaluate per-frame and weight by valid pixels.

        If align_scale_only=True, perform a single per-SEQUENCE scale-only
        alignment (one global s, no shift), then evaluate per-frame and weight
        by valid pixels.

        If align_single_frame=True, align each frame independently with median
        scale and aggregate metrics weighted by valid pixels (MonST3R
        single-frame protocol style).

        Otherwise (align_median=True), align each frame independently with
        per-frame median scale (standard Eigen protocol).
        """
        def _valid_mask(gt, pred=None):
            v = gt > 0
            if max_depth is not None:
                v &= (gt < max_depth)
            if use_eigen_crop:
                v &= _eigen_crop_mask(gt.shape)
            # Skip zero-padded regions (e.g. from center_crop preprocessing)
            if pred is not None:
                v &= (pred > 0)
            return v

        # ── Resize all preds to match GT resolution ───────────────────────
        pred_list_r = []
        for pred, gt in zip(pred_list, gt_list):
            if pred.shape != gt.shape:
                pred = _resize_depth(pred, gt.shape)
            pred_list_r.append(pred)

        if align_scale_shift:
            # MonST3R protocol: stack ALL valid pixels across all frames,
            # find one global (s, t) for the whole sequence, then evaluate.
            all_pred = np.concatenate([p.ravel() for p in pred_list_r])
            all_gt   = np.concatenate([g.ravel() for g in gt_list])
            all_valid = np.concatenate([_valid_mask(g, p).ravel() for p, g in zip(pred_list_r, gt_list)])
            # Compute global scale+shift on all valid pixels
            p_valid = all_pred[all_valid]
            g_valid = all_gt[all_valid]
            s_init = float(np.median(g_valid) / (np.median(p_valid) + 1e-8))
            # Run Adam L1 minimisation (delegates to metrics.scale_shift_align)
            # We need s, t — call on a dummy array to extract coefficients
            # Use a direct per-pixel aligned version
            import torch
            p_t = torch.tensor(p_valid, dtype=torch.float32)
            g_t = torch.tensor(g_valid, dtype=torch.float32)
            s_p = torch.tensor([s_init], requires_grad=True, dtype=torch.float32)
            t_p = torch.tensor([0.0],    requires_grad=True, dtype=torch.float32)
            opt = torch.optim.Adam([s_p, t_p], lr=1e-4)
            prev = None
            for _ in range(1000):
                opt.zero_grad()
                loss = torch.abs(s_p * p_t + t_p - g_t).sum()
                loss.backward(); opt.step()
                cur = loss.item()
                if prev is not None and abs(prev - cur) < 1e-6:
                    break
                prev = cur
            s_val = float(s_p.detach())
            t_val = float(t_p.detach())
            print(f"[eval:kitti] MonST3R scale+shift: s={s_val:.4f}, t={t_val:.4f}")

            # Now evaluate per-frame with the global (s, t)
            accum   = {k: [] for k in ("abs_rel","sq_rel","rmse","rmse_log","d1","d2","d3")}
            weights = []
            for pred, gt in zip(pred_list_r, gt_list):
                valid = _valid_mask(gt, pred)
                if valid.sum() == 0:
                    continue
                pred_v = (pred[valid].astype(np.float64) * s_val + t_val)
                gt_v   = gt[valid].astype(np.float64)
                pred_v = np.clip(pred_v, 1e-5, 1e9)
                m = depth_metrics(pred_v, gt_v)
                n = int(valid.sum())
                for k in accum:
                    accum[k].append(m[k])
                weights.append(n)
            if not accum["abs_rel"]:
                raise RuntimeError("No valid frames evaluated")
            # Weighted mean by valid_pixels (MonST3R aggregation)
            w = np.array(weights, dtype=np.float64)
            out = {k: float(np.average(accum[k], weights=w)) for k in accum}
            out["valid_pixels"] = int(np.sum(weights))
            return out

        if align_scale_only:
            # Per-sequence scale-only: stack all valid pixels, fit one global s.
            all_pred = np.concatenate([p.ravel() for p in pred_list_r])
            all_gt   = np.concatenate([g.ravel() for g in gt_list])
            all_valid = np.concatenate([_valid_mask(g, p).ravel() for p, g in zip(pred_list_r, gt_list)])
            p_valid = all_pred[all_valid]
            g_valid = all_gt[all_valid]
            s_val = scale_only_fit(p_valid, g_valid)
            print(f"[eval:kitti] scale-only factor: s={s_val:.4f}")

            accum   = {k: [] for k in ("abs_rel","sq_rel","rmse","rmse_log","d1","d2","d3")}
            weights = []
            for pred, gt in zip(pred_list_r, gt_list):
                valid = _valid_mask(gt, pred)
                if valid.sum() == 0:
                    continue
                pred_v = pred[valid].astype(np.float64) * s_val
                gt_v   = gt[valid].astype(np.float64)
                pred_v = np.clip(pred_v, 1e-5, 1e9)
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

        else:
            # Standard per-frame alignment
            accum = {k: [] for k in ("abs_rel","sq_rel","rmse","rmse_log","d1","d2","d3")}
            weights = []
            for pred, gt in zip(pred_list_r, gt_list):
                valid = _valid_mask(gt, pred)
                if valid.sum() == 0:
                    continue
                pred_v = pred[valid].astype(np.float64)
                gt_v   = gt[valid].astype(np.float64)
                pred_v = np.clip(pred_v, 1e-3, max_depth if max_depth else 1e9)
                if align_median:
                    scale  = np.median(gt_v) / (np.median(pred_v) + 1e-8)
                    pred_v = pred_v * scale
                pred_v = np.clip(pred_v, 1e-3, max_depth if max_depth else 1e9)
                m = depth_metrics(pred_v, gt_v)
                for k in accum:
                    accum[k].append(m[k])
                weights.append(int(valid.sum()))
            if not accum["abs_rel"]:
                raise RuntimeError("No valid frames evaluated")
            if align_single_frame:
                # MonST3R single-frame protocol: weighted by valid pixels.
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
        """Load depth/*.npy from output_dir, sorted by filename.

        Prefers depth_orig_res/ (letterbox-corrected, original image resolution)
        over depth/ (raw VGGT 518×518 space) when available.
        """
        # Prefer original-resolution depths (letterbox-corrected)
        for candidate in ("depth_orig_res", "depth"):
            depth_dir = os.path.join(output_dir, candidate)
            if os.path.isdir(depth_dir):
                files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".npy"))
                if files:
                    if candidate == "depth_orig_res":
                        print("[eval:kitti] using depth_orig_res/ (letterbox-corrected)")
                    return [np.load(os.path.join(depth_dir, f)).astype(np.float32)
                            for f in files]
        # Final fallback: .npy directly in output_dir
        files = sorted(f for f in os.listdir(output_dir) if f.endswith(".npy"))
        if not files:
            raise FileNotFoundError(f"No depth .npy files found in {output_dir}")
        return [np.load(os.path.join(output_dir, f)).astype(np.float32)
                for f in files]

    @staticmethod
    def _load_gt_depths(gt_dir: str, n_frames: int,
                        drive: str = None) -> list:
        """Load GT depth from gt_dir.

        Supports:
            - .npy  files (float32, metres)
            - .png  files (uint16, value / 256 = metres)

        If ``drive`` is given (e.g. '2011_09_26_drive_0002_sync'), only files
        whose name contains that string are loaded.  This is needed for the
        KITTI val_selection_cropped layout where all drives share one folder.

        Files are sorted; must match n_frames after filtering.
        """
        if not os.path.isdir(gt_dir):
            raise FileNotFoundError(f"GT depth directory not found: {gt_dir}")

        all_files = sorted(os.listdir(gt_dir))

        # Filter by drive name if given
        if drive:
            all_files = [f for f in all_files if drive in f]
            if not all_files:
                raise FileNotFoundError(
                    f"No GT files matching drive '{drive}' in {gt_dir}"
                )
            print(f"[eval:kitti] filtered to {len(all_files)} GT files for drive '{drive}'")

        npy_files = [f for f in all_files if f.endswith(".npy")]
        png_files = [f for f in all_files if f.endswith(".png")]

        if npy_files:
            files  = npy_files
            loader = lambda p: np.load(p).astype(np.float32)
        elif png_files:
            files  = png_files
            loader = _load_kitti_depth_png
        else:
            raise FileNotFoundError(f"No .npy or .png GT depth files in {gt_dir}")

        if len(files) != n_frames:
            print(
                f"[eval:kitti] WARNING: GT has {len(files)} frames but pred has "
                f"{n_frames}.  Using min({len(files)}, {n_frames}) frames."
            )
            files = files[: min(len(files), n_frames)]

        return [loader(os.path.join(gt_dir, f)) for f in files]

    # ------------------------------------------------------------------
    # Side-by-side comparison
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
# standalone helpers
# ---------------------------------------------------------------------------

def _load_kitti_depth_png(path: str) -> np.ndarray:
    """Load KITTI uint16 depth PNG → float32 metres (0 = invalid)."""
    try:
        from PIL import Image
        img = np.array(Image.open(path), dtype=np.float32)
    except ImportError:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
    return img / 256.0   # uint16 encoding: metres * 256


def _eigen_crop_mask(shape) -> np.ndarray:
    """Return boolean mask for the Eigen crop region."""
    H, W  = shape
    mask  = np.zeros(shape, dtype=bool)
    r0    = int(H * KITTIEvaluator._CROP_T)
    r1    = int(H * KITTIEvaluator._CROP_B)
    c0    = int(W * KITTIEvaluator._CROP_L)
    c1    = int(W * KITTIEvaluator._CROP_R)
    mask[r0:r1, c0:c1] = True
    return mask


def _resize_depth(depth: np.ndarray, target_shape) -> np.ndarray:
    """Bilinear resize depth map to target (H, W)."""
    try:
        import cv2
        return cv2.resize(depth, (target_shape[1], target_shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        img = Image.fromarray(depth).resize(
            (target_shape[1], target_shape[0]), Image.BILINEAR
        )
        return np.array(img, dtype=np.float32)
