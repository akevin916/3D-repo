#!/usr/bin/env python3
"""
visualize.py — Side-by-side diagnostic GIF for vggt-dyn run.py outputs.

Layout (2x3 panel grid):
    [ RGB | Depth (Turbo) | Dynamic Mask overlay ]
    [ Depth Conf | RAFT Flow | Ego-Flow ]

--raft is optional:
    - not given  -> RAFT Flow / Ego-Flow panels show "N/A (no --raft)"
    - given      -> RAFT flow + ego-flow (from VGGT depth/pose) + residual
                    are computed, and a recomputed dynamic mask (via
                    --threshold) is reported in stats.json

Image source:
    --images is optional. If omitted, the original frame paths are read
    from <output_dir>/metrics.json["image_paths"] (relative paths, written
    by run.py since the 2026-06 pipeline update).

Usage:
    python visualize.py --output_dir outputs/my_scene
    python visualize.py --output_dir outputs/my_scene --raft third_party/weights/raft/raft-things.pth

Output:
    <output_dir>/viz.gif
    <output_dir>/stats.json   (only meaningful fields filled when --raft given)
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import cv2
import imageio.v2 as imageio
import torch

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

_VGGT_REPO = os.path.join(_PROJECT_DIR, "vggt")
if _VGGT_REPO not in sys.path:
    sys.path.insert(0, _VGGT_REPO)


# ---------------------------------------------------------------------------
# Colormap / panel helpers
# ---------------------------------------------------------------------------

def depth_to_colormap(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """[H, W] float -> [H, W, 3] BGR uint8 via Turbo colormap."""
    d = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    d_u8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)


def conf_to_colormap(conf: np.ndarray) -> np.ndarray:
    """[H, W] float (any range) -> [H, W, 3] BGR via Viridis colormap."""
    mn, mx = float(conf.min()), float(conf.max())
    norm = (conf - mn) / (mx - mn + 1e-8)
    u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)


def flow_to_color(flow: np.ndarray, max_mag: float = None) -> np.ndarray:
    """[2, H, W] float flow -> [H, W, 3] BGR via HSV colorwheel.

    Hue = direction, Value = magnitude (clipped to max_mag).
    """
    fx, fy = flow[0], flow[1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx)  # [-pi, pi]

    if max_mag is None or max_mag <= 0:
        max_mag = float(np.percentile(mag, 99)) + 1e-6

    h = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    s = np.full_like(h, 255)
    v = (np.clip(mag / max_mag, 0, 1) * 255).astype(np.uint8)

    hsv = np.stack([h, s, v], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def residual_to_colormap(residual: np.ndarray) -> np.ndarray:
    """[H, W] float -> [H, W, 3] BGR via HOT colormap (blue=low, red=high)."""
    p99 = float(np.percentile(residual, 99)) + 1e-6
    norm = np.clip(residual / p99, 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_HOT)


def mask_overlay(rgb_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlay a bool [H, W] mask as semi-transparent red on a BGR image."""
    out = rgb_bgr.copy()
    if mask.any():
        red = np.zeros_like(out)
        red[:, :, 2] = 255
        out[mask] = (out[mask] * (1.0 - alpha) + red[mask] * alpha).astype(np.uint8)
    return out


def add_label(img_bgr: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    """Burn a label into the top-left corner of a BGR image."""
    out = img_bgr.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return out


def na_panel(H: int, W: int, text: str) -> np.ndarray:
    """Gray placeholder panel with centered text."""
    out = np.full((H, W, 3), 40, dtype=np.uint8)
    (tw, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    org = (max(0, (W - tw) // 2), max(0, (H + th_) // 2))
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    return out


def compute_residual_stats(residual: np.ndarray, mask: np.ndarray = None) -> dict:
    """[H, W] residual -> stats dict."""
    r = residual[mask] if mask is not None and mask.any() else residual
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {}
    return {
        "mean":   float(r.mean()),
        "median": float(np.median(r)),
        "p95":    float(np.percentile(r, 95)),
        "max":    float(r.max()),
    }


# ---------------------------------------------------------------------------
# Image preprocessing for RAFT (must match VGGT's pixel space)
# ---------------------------------------------------------------------------

def preprocess_images_for_raft(image_paths, preprocess_mode: str, H: int, W: int) -> np.ndarray:
    """Return [S, H, W, 3] float32 images in [0,1], in the same pixel space
    as the VGGT depth/pose outputs (center_crop / long_edge / letterbox)."""
    from PIL import Image as PILImage
    import torchvision.transforms.functional as TF

    imgs = []
    if preprocess_mode == "center_crop":
        for p in image_paths:
            img = PILImage.open(p).convert("RGB")
            w, h = img.size
            sq = min(w, h)
            left, top = (w - sq) // 2, (h - sq) // 2
            img = img.crop((left, top, left + sq, top + sq))
            img = img.resize((H, H), PILImage.Resampling.BICUBIC)
            imgs.append(TF.to_tensor(img))
    elif preprocess_mode == "long_edge":
        target_long = max(H, W)
        for p in image_paths:
            img = PILImage.open(p).convert("RGB")
            w, h = img.size
            if w >= h:
                new_w = target_long
                new_h = round(h * (new_w / w) / 14) * 14
            else:
                new_h = target_long
                new_w = round(w * (new_h / h) / 14) * 14
            new_w, new_h = max(new_w, 14), max(new_h, 14)
            img = img.resize((new_w, new_h), PILImage.Resampling.BICUBIC)
            img_t = TF.to_tensor(img)
            if img_t.shape[1] != H or img_t.shape[2] != W:
                img_t = TF.resize(img_t, [H, W],
                                   interpolation=TF.InterpolationMode.BILINEAR,
                                   antialias=True)
            imgs.append(img_t)
    else:  # letterbox
        from vggt.utils.load_fn import load_and_preprocess_images_square
        imgs_tensor, _ = load_and_preprocess_images_square(image_paths, target_size=H)
        return imgs_tensor.permute(0, 2, 3, 1).numpy().astype(np.float32)

    return torch.stack(imgs).permute(0, 2, 3, 1).numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--output_dir", required=True,
                   help="run.py --output directory")
    p.add_argument("--images", default=None,
                   help='Glob pattern for original input images. If omitted, '
                        'read from <output_dir>/metrics.json["image_paths"].')
    p.add_argument("--raft", default=None,
                   help="RAFT checkpoint (.pth). If omitted, flow panels show N/A.")
    p.add_argument("--threshold", type=float, default=0.35,
                   help="flow-residual threshold for recomputed dynamic mask stats")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--fps", type=int, default=5, help="GIF playback fps")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk_size", type=int, default=8)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    out_dir = args.output_dir

    # ── 1. Resolve image paths ────────────────────────────────────────────────
    metrics_path = os.path.join(out_dir, "metrics.json")
    metrics = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    if args.images:
        image_paths = sorted(glob.glob(args.images))
        if not image_paths:
            raise FileNotFoundError(f"No images found: {args.images}")
    else:
        rel_paths = metrics.get("image_paths")
        if not rel_paths:
            raise ValueError(
                f"No --images given and {metrics_path} has no 'image_paths' "
                "(re-run run.py to regenerate, or pass --images explicitly)."
            )
        image_paths = [os.path.normpath(os.path.join(out_dir, p)) for p in rel_paths]

    # ── 2. Load run.py outputs ────────────────────────────────────────────────
    depth_paths = sorted(glob.glob(os.path.join(out_dir, "depth", "*.npy")))
    mask_paths  = sorted(glob.glob(os.path.join(out_dir, "dynamic_mask", "*.npy")))
    conf_paths  = sorted(glob.glob(os.path.join(out_dir, "depth_conf", "*.npy")))
    if not depth_paths:
        raise FileNotFoundError(f"No depth/*.npy found in {out_dir}")

    depth = np.stack([np.load(p) for p in depth_paths], axis=0)         # [S,H,W]
    extrinsics = np.load(os.path.join(out_dir, "extrinsics.npy"))       # [S,3,4]
    intrinsics = np.load(os.path.join(out_dir, "intrinsics.npy"))       # [S,3,3]

    if mask_paths:
        dyn_mask_pre = np.stack([np.load(p).astype(bool) for p in mask_paths], axis=0)
    else:
        dyn_mask_pre = np.zeros(depth.shape, dtype=bool)

    has_conf = len(conf_paths) == depth.shape[0]
    depth_conf = np.stack([np.load(p) for p in conf_paths], axis=0) if has_conf else None
    if not has_conf:
        print(f"[viz] no depth_conf/*.npy found in {out_dir} — Conf panel will show N/A")

    # ── 3. Align frame counts (images vs saved outputs) ──────────────────────
    S = depth.shape[0]
    if args.max_frames is not None:
        S = min(S, args.max_frames)
    if len(image_paths) != S:
        print(f"[viz] {len(image_paths)} images vs {S} saved frames — using min")
        S = min(S, len(image_paths))
    image_paths = image_paths[:S]
    depth, extrinsics, intrinsics, dyn_mask_pre = (
        depth[:S], extrinsics[:S], intrinsics[:S], dyn_mask_pre[:S]
    )
    if depth_conf is not None:
        depth_conf = depth_conf[:S]

    H, W = depth.shape[1], depth.shape[2]
    print(f"[viz] {S} frames, depth shape ({H}, {W})")

    # ── 4. Load RGB (resized to depth resolution) ────────────────────────────
    images_bgr = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        images_bgr.append(cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR))

    # ── 5. Depth colormap range (global percentile) ──────────────────────────
    all_vals = depth.ravel()
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))
    print(f"[viz] depth range (2-98 pct): [{vmin:.4f}, {vmax:.4f}]")

    # ── 6. Optional flow/conf analysis ────────────────────────────────────────
    flow_fwd = ego_flow = residual = valid_fwd = None
    dyn_mask_computed = None
    max_flow_mag = None

    if args.raft:
        preprocess_mode = metrics.get("preprocess", "letterbox")
        print(f"[viz] preprocess mode: {preprocess_mode}")

        print(f"[viz] preprocessing images for RAFT ({preprocess_mode}) ...")
        images_np = preprocess_images_for_raft(image_paths, preprocess_mode, H, W)

        print("[viz] computing RAFT flow ...")
        from vggt_dyn.utils.flow_utils import compute_adjacent_flow
        flow_fwd_t, _, valid_fwd_t, _ = compute_adjacent_flow(
            list(images_np), model_path=args.raft,
            chunk_size=args.chunk_size, device=device_str,
        )
        flow_fwd = flow_fwd_t.cpu().numpy()    # [S-1, 2, H, W]
        valid_fwd = valid_fwd_t.cpu().numpy()  # [S-1, 1, H, W]

        print("[viz] computing ego-flow from VGGT depth + pose ...")
        from vggt_dyn.dynamic_mask import compute_ego_flow, build_dynamic_mask, pair_mask_to_frame_mask
        d_t = torch.from_numpy(depth).float().to(device)
        R_t = torch.from_numpy(extrinsics[:, :3, :3]).float().to(device)
        T_t = torch.from_numpy(extrinsics[:, :3, 3]).float().to(device)
        K_t = torch.from_numpy(intrinsics).float().to(device)
        ego_flow = compute_ego_flow(d_t, R_t, T_t, K_t).cpu().numpy()  # [S-1, 2, H, W]

        diff = ego_flow - flow_fwd
        residual = np.sqrt((diff ** 2).sum(axis=1))         # [S-1, H, W]
        valid_np = valid_fwd[:, 0, :, :].astype(bool)
        residual[~valid_np] = 0.0

        resid_t = torch.from_numpy(residual).unsqueeze(1)
        pair_mask = build_dynamic_mask(resid_t, threshold=args.threshold, normalize=True)
        frame_mask = pair_mask_to_frame_mask(pair_mask, S)
        dyn_mask_computed = frame_mask[:, 0].numpy()  # [S, H, W]

        if dyn_mask_pre.any() or dyn_mask_computed.any():
            match = float(np.mean(dyn_mask_pre == dyn_mask_computed)) * 100
            print(f"[viz] pre-saved mask vs recomputed (thr={args.threshold:.2f}): "
                  f"pixel agreement = {match:.1f}%")

        max_flow_mag = float(np.percentile(np.sqrt((flow_fwd ** 2).sum(axis=1)), 99))

    # ── 7. Per-frame panels ────────────────────────────────────────────────────
    print(f"[viz] generating {S} panels ...")
    all_stats = []
    gif_frames = []

    for i in range(S):
        rgb_bgr = images_bgr[i]
        dm = dyn_mask_computed[i] if dyn_mask_computed is not None else dyn_mask_pre[i]

        p_rgb   = add_label(rgb_bgr, f"RGB [{i:04d}]")
        p_depth = add_label(depth_to_colormap(depth[i], vmin, vmax), "Depth")
        p_mask  = add_label(mask_overlay(rgb_bgr, dm), f"Dyn Mask {dm.mean()*100:.1f}%",
                             color=(0, 80, 255))

        if depth_conf is not None:
            c = depth_conf[i]
            p_conf = add_label(conf_to_colormap(c),
                                f"Conf [{c.min():.3f},{c.max():.3f}]")
        else:
            p_conf = na_panel(H, W, "Conf: N/A")

        frame_stats = {"frame": i, "dyn_pct": round(float(dm.mean()) * 100, 2)}
        if depth_conf is not None:
            frame_stats["conf_mean"] = float(depth_conf[i].mean())
            frame_stats["conf_min"]  = float(depth_conf[i].min())
            frame_stats["conf_max"]  = float(depth_conf[i].max())

        if flow_fwd is not None:
            pair_idx = min(i, S - 2)
            rf, ef, rs = flow_fwd[pair_idx], ego_flow[pair_idx], residual[pair_idx]
            p_raft = add_label(flow_to_color(rf, max_flow_mag), "RAFT Flow")
            p_ego  = add_label(flow_to_color(ef, max_flow_mag), "Ego-Flow")
            r_stats = compute_residual_stats(rs, valid_fwd[pair_idx, 0].astype(bool))
            frame_stats["raft_mean"] = float(np.sqrt((rf ** 2).sum(axis=0)).mean())
            frame_stats["ego_mean"]  = float(np.sqrt((ef ** 2).sum(axis=0)).mean())
            frame_stats["resid_median"] = r_stats.get("median", float("nan"))
            frame_stats["resid_p95"]    = r_stats.get("p95", float("nan"))
        else:
            p_raft = na_panel(H, W, "RAFT Flow: N/A (no --raft)")
            p_ego  = na_panel(H, W, "Ego-Flow: N/A (no --raft)")

        all_stats.append(frame_stats)

        row0 = np.concatenate([p_rgb, p_depth, p_mask], axis=1)
        row1 = np.concatenate([p_conf, p_raft, p_ego], axis=1)
        panel = np.concatenate([row0, row1], axis=0)
        gif_frames.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))

    # ── 8. Save outputs ────────────────────────────────────────────────────────
    gif_path = os.path.join(out_dir, "viz.gif")
    duration = 1.0 / args.fps
    imageio.mimsave(gif_path, gif_frames, duration=duration, loop=0)
    print(f"[viz] saved → {gif_path}")

    stats_path = os.path.join(out_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"[viz] stats saved → {stats_path}")

    print(f"[viz] layout: RGB | Depth | Dyn Mask // Conf | RAFT Flow | Ego-Flow  "
          f"({S} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
