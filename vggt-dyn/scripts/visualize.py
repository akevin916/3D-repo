#!/usr/bin/env python3
"""
visualize.py — Generate a side-by-side visualization video from vggt-dyn outputs.

Layout per frame:
  [ Original RGB | Depth (Turbo colormap) | Dynamic Mask overlay (red) ]

Usage:
    python visualize.py \
        --output_dir /tmp/vggt_dyn_smoke \
        --images     "/tmp/smoke_frames/*.png" \
        --fps        10

Output: <output_dir>/viz.mp4
"""

import argparse
import glob
import os

import cv2
import imageio
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def depth_to_colormap(
    depth: np.ndarray,       # [H, W] float32
    vmin: float,
    vmax: float,
) -> np.ndarray:             # [H, W, 3] BGR uint8
    """Normalize depth to [0,1] with global percentile range, then apply Turbo colormap."""
    d = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    d_u8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)


def mask_overlay(
    rgb_bgr: np.ndarray,     # [H, W, 3] BGR uint8
    mask: np.ndarray,        # [H, W] bool
    alpha: float = 0.55,
) -> np.ndarray:             # [H, W, 3] BGR uint8
    """Overlay dynamic-pixel mask as a semi-transparent red tint."""
    out = rgb_bgr.copy()
    if mask.any():
        red = np.zeros_like(out)
        red[:, :, 2] = 255  # BGR → red channel
        out[mask] = (out[mask] * (1.0 - alpha) + red[mask] * alpha).astype(np.uint8)
    # Draw a thin border so the panel is distinguishable
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 200), 2)
    return out


def add_label(img_bgr: np.ndarray, text: str) -> np.ndarray:
    """Burn a label into the top-left corner of a BGR image."""
    out = img_bgr.copy()
    cv2.putText(
        out, text,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 0), 3, cv2.LINE_AA,   # black outline
    )
    cv2.putText(
        out, text,
        (8, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 1, cv2.LINE_AA,  # white text
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize vggt-dyn results as a side-by-side video"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="vggt-dyn run output directory (contains depth/ and dynamic_mask/)",
    )
    parser.add_argument(
        "--images", required=True,
        help='Glob pattern for original input images, same as run.py --images. '
             'Wrap in quotes: "frames/*.png"',
    )
    parser.add_argument("--fps",  type=int,   default=10,   help="Output video FPS (default: 10)")
    parser.add_argument("--out",  default=None,             help="Output path (default: <output_dir>/viz.mp4)")
    parser.add_argument("--gif",  action="store_true",       help="Also save an animated GIF (viewable in VS Code)")
    parser.add_argument("--p_lo", type=float, default=2.0,  help="Lower depth percentile for colormap (default: 2)")
    parser.add_argument("--p_hi", type=float, default=98.0, help="Upper depth percentile for colormap (default: 98)")
    args = parser.parse_args()

    out_path    = args.out or os.path.join(args.output_dir, "viz.mp4")
    depth_dir   = os.path.join(args.output_dir, "depth")
    mask_dir    = os.path.join(args.output_dir, "dynamic_mask")

    # ── Gather files ─────────────────────────────────────────────────────────
    image_paths = sorted(glob.glob(args.images))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
    mask_files  = sorted(glob.glob(os.path.join(mask_dir,  "*.npy")))

    if not image_paths:
        raise FileNotFoundError(f"No images found: {args.images}")

    S = len(image_paths)
    if len(depth_files) == 0:
        raise FileNotFoundError(f"No depth .npy files found in {depth_dir}")
    if len(depth_files) != S:
        print(f"[warn] {len(depth_files)} depth files vs {S} images — using min({S}, {len(depth_files)})")
        S = min(S, len(depth_files))
        image_paths = image_paths[:S]
        depth_files = depth_files[:S]
        mask_files  = mask_files[:S] if mask_files else []

    has_masks = len(mask_files) == S
    if not has_masks:
        print(f"[warn] no mask files found in {mask_dir}, skipping mask overlay")

    # ── Load depths & compute global percentile range ─────────────────────────
    print(f"[viz] loading {S} depth maps ...")
    depths = [np.load(f).astype(np.float32) for f in depth_files]
    all_vals = np.concatenate([d.ravel() for d in depths])
    vmin = float(np.percentile(all_vals, args.p_lo))
    vmax = float(np.percentile(all_vals, args.p_hi))
    print(f"[viz] depth range ({args.p_lo:.0f}–{args.p_hi:.0f} pct): [{vmin:.4f}, {vmax:.4f}]")

    # ── Determine output frame size from depth ────────────────────────────────
    H, W = depths[0].shape[:2]

    # ── Initialise video writer ───────────────────────────────────────────────
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (W * 3, H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open: {out_path}")
    gif_frames = [] if args.gif else None

    # ── Write frames ──────────────────────────────────────────────────────────
    print(f"[viz] writing {S} frames → {out_path}")
    for i in range(S):
        # Panel 1 — original RGB
        img_bgr = cv2.imread(image_paths[i])
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_paths[i]}")
        img_bgr = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

        # Panel 2 — depth colormap
        depth_bgr = depth_to_colormap(depths[i], vmin, vmax)

        # Panel 3 — dynamic mask overlay
        if has_masks:
            mask = np.load(mask_files[i]).squeeze().astype(bool)  # [H, W]
            panel3 = mask_overlay(img_bgr, mask)
        else:
            panel3 = img_bgr.copy()

        # Labels
        p1 = add_label(img_bgr,   f"RGB  [{i:04d}]")
        p2 = add_label(depth_bgr, "Depth")
        p3 = add_label(panel3,    "Dynamic Mask")

        frame = np.concatenate([p1, p2, p3], axis=1)
        writer.write(frame)
        if gif_frames is not None:
            gif_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    writer.release()

    # ── Save GIF ──────────────────────────────────────────────────────────────
    if gif_frames is not None:
        gif_path = out_path.rsplit(".", 1)[0] + ".gif"
        duration = 1.0 / args.fps
        imageio.mimsave(gif_path, gif_frames, duration=duration, loop=0)
        print(f"[viz] GIF  → {gif_path}  (open in VS Code explorer to preview)")
    dyn_pct = 0.0
    if has_masks:
        masks_all = [np.load(f).squeeze().astype(bool) for f in mask_files]
        dyn_pct = float(np.mean([m.mean() for m in masks_all])) * 100
    print(f"[viz] done  — avg dynamic fraction: {dyn_pct:.1f}%")
    print(f"[viz] saved → {out_path}")
    print(f"      layout: RGB | Depth (Turbo) | Dynamic Mask overlay  ({S} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
