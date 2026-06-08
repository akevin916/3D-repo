"""evaluators/sintel.py — MPI-Sintel depth evaluator.

Evaluation purpose: monocular depth estimation on Sintel dynamic scenes.

Dataset structure (common):
    <scene>/
        frame_0001.dpt
        frame_0002.dpt
        ...

Input (from run.py --output):
    depth/XXXX.npy or depth_orig_res/XXXX.npy  per-frame predicted depth [H, W]

Metrics:
    AbsRel, SqRel, RMSE, RMSElog, δ1 / δ2 / δ3
"""

import os
import numpy as np

from .base import BaseEvaluator
from .metrics import depth_metrics, print_metrics, scale_only_fit


SINTEL_DPT_TAG = 202021.25


class SintelEvaluator(BaseEvaluator):
    """Evaluates depth outputs on MPI-Sintel."""

    def run(self, args) -> dict:
        pred_depths = self._load_pred_depths(args.output_dir)
        gt_depths = self._load_gt_depths(args.scene_dir, len(pred_depths))

        if len(pred_depths) != len(gt_depths):
            raise ValueError(
                f"Frame count mismatch: pred={len(pred_depths)}, gt={len(gt_depths)}"
            )

        scene_name = os.path.basename(args.scene_dir.rstrip("/"))
        print(f"[eval:sintel] evaluating {len(pred_depths)} frames (scene: {scene_name}) ...")

        mode = args.align_scale_mode
        max_depth = args.max_depth if args.max_depth > 0 else None

        if mode == "scale_and_shift":
            print("[eval:sintel] alignment: scale+shift (MonST3R protocol: align_with_lad2)")
        elif mode == "scale_only":
            print("[eval:sintel] alignment: per-sequence scale only (no shift)")
        elif mode == "single_frame":
            print("[eval:sintel] alignment: single-frame median scale (MonST3R single-frame protocol)")
        elif mode == "median":
            print("[eval:sintel] alignment: per-frame median scale")
        else:
            print("[eval:sintel] alignment: none")

        if max_depth is None:
            print("[eval:sintel] depth cap: none")
        else:
            print(f"[eval:sintel] depth cap: {max_depth} m")

        if args.post_clip_max > 0:
            print(f"[eval:sintel] post-clip max: {args.post_clip_max} m")

        results = self._evaluate_sequence(
            pred_depths,
            gt_depths,
            align_median=(mode in ("median", "single_frame")),
            align_single_frame=(mode == "single_frame"),
            align_scale_only=(mode == "scale_only"),
            align_scale_shift=(mode == "scale_and_shift"),
            max_depth=max_depth,
            min_depth=args.min_depth,
            post_clip_max=args.post_clip_max if args.post_clip_max > 0 else None,
        )

        if "d1" in results:
            results["delta_lt_1_25"] = float(results["d1"])
        if "d2" in results:
            results["delta_lt_1_25_sq"] = float(results["d2"])
        if "d3" in results:
            results["delta_lt_1_25_cu"] = float(results["d3"])

        title = f"Sintel Depth — VGGT-Dyn  [{scene_name}]"
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
            suffix = f"sintel_{scene_name}_scale_shift"
        elif mode == "scale_only":
            suffix = f"sintel_{scene_name}_scale_only"
        elif mode == "single_frame":
            suffix = f"sintel_{scene_name}_single_frame"
        elif mode == "median":
            suffix = f"sintel_{scene_name}_median"
        else:
            suffix = f"sintel_{scene_name}"
        path = self.save_results(results, args.output_dir, suffix)
        print(f"[eval:sintel] saved -> {path}")
        return results

    @staticmethod
    def _evaluate_sequence(
        pred_list,
        gt_list,
        align_median: bool,
        align_single_frame: bool,
        align_scale_only: bool,
        align_scale_shift: bool,
        max_depth=70.0,
        min_depth=0.0,
        post_clip_max=None,
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
            if post_clip_max is not None:
                pred = np.minimum(pred, post_clip_max)
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
            print(f"[eval:sintel] MonST3R scale+shift: s={s_val:.4f}, t={t_val:.4f}")

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
            print(f"[eval:sintel] scale-only factor: s={s_val:.4f}")

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

    def _load_pred_depths(self, output_dir: str) -> list:
        for candidate in ("depth_orig_res", "depth"):
            depth_dir = os.path.join(output_dir, candidate)
            if os.path.isdir(depth_dir):
                files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".npy"))
                if files:
                    if candidate == "depth_orig_res":
                        print("[eval:sintel] using depth_orig_res/ (letterbox-corrected)")
                    return [
                        np.load(os.path.join(depth_dir, f)).astype(np.float32)
                        for f in files
                    ]

        files = sorted(f for f in os.listdir(output_dir) if f.endswith(".npy"))
        if not files:
            raise FileNotFoundError(f"No depth .npy files found in {output_dir}")
        return [np.load(os.path.join(output_dir, f)).astype(np.float32) for f in files]

    @staticmethod
    def _load_gt_depths(scene_dir: str, n_frames: int) -> list:
        depth_dir = os.path.join(scene_dir, "depth")
        scan_dir = depth_dir if os.path.isdir(depth_dir) else scene_dir

        files = [
            os.path.join(scan_dir, f)
            for f in sorted(os.listdir(scan_dir))
            if f.endswith(".dpt")
        ]

        if not files:
            raise FileNotFoundError(f"No GT .dpt files found in {scan_dir}")

        if len(files) != n_frames:
            print(
                f"[eval:sintel] WARNING: GT has {len(files)} frames, pred has "
                f"{n_frames}. Using first {min(len(files), n_frames)}."
            )
            files = files[: min(len(files), n_frames)]

        return [_read_sintel_dpt(p) for p in files]


def _read_sintel_dpt(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        tag = np.fromfile(f, np.float32, count=1)[0]
        if abs(float(tag) - SINTEL_DPT_TAG) > 1e-4:
            raise ValueError(f"Invalid Sintel .dpt tag in {path}: {tag}")
        width = int(np.fromfile(f, np.int32, count=1)[0])
        height = int(np.fromfile(f, np.int32, count=1)[0])
        data = np.fromfile(f, np.float32, count=width * height)
    if data.size != width * height:
        raise ValueError(f"Corrupted Sintel .dpt file: {path}")
    depth = data.reshape((height, width)).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0] = 0.0
    return depth


def _resize_depth(depth: np.ndarray, target_shape) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(
            depth,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    except ImportError:
        from PIL import Image
        return np.array(
            Image.fromarray(depth).resize(
                (target_shape[1], target_shape[0]),
                Image.BILINEAR,
            ),
            dtype=np.float32,
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
