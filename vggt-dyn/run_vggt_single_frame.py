#!/usr/bin/env python3
"""Run original VGGT depth inference in single-frame mode.

This script performs pure feed-forward VGGT prediction per image (S=1 each
forward pass), without any test-time optimization, then saves outputs in a
layout compatible with vggt-dyn/eval.py:

    <output>/depth/0000.npy
    <output>/depth_orig_res/0000.npy

Usage example:
    python run_vggt_single_frame.py \
      --images "../data/bonn/rgbd_bonn_dataset/rgbd_bonn_balloon2/rgb_110/*.png" \
      --checkpoint ../vggt/checkpoints/VGGT-1B.pt \
      --output outputs/vggt_single/bonn_balloon2
"""

import os
import glob
import json
import argparse
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VGGT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "vggt"))

import sys
if VGGT_ROOT not in sys.path:
    sys.path.insert(0, VGGT_ROOT)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square


def _crop_depth_to_orig(depth_hw: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Map 518x518 depth back to original image size using letterbox coords."""
    import cv2

    x1, y1, x2, y2, orig_w, orig_h = coords
    orig_w, orig_h = int(round(orig_w)), int(round(orig_h))

    r1, r2 = int(round(y1)), int(round(y2))
    c1, c2 = int(round(x1)), int(round(x2))
    h, w = depth_hw.shape
    r1, r2 = max(0, r1), min(h, r2)
    c1, c2 = max(0, c1), min(w, c2)

    crop = depth_hw[r1:r2, c1:c2]
    if crop.size == 0:
        crop = depth_hw

    out = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float32)


def _load_model(checkpoint: str, device: str) -> VGGT:
    model = VGGT().to(device)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Original VGGT single-frame depth inference")
    p.add_argument("--images", required=True, help="glob pattern for input frames")
    p.add_argument("--checkpoint", default="../vggt/checkpoints/VGGT-1B.pt",
                   help="path to VGGT checkpoint")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--max_frames", type=int, default=None, help="optional cap on frame count")
    p.add_argument("--save_conf", action="store_true", help="also save depth confidence maps")
    return p.parse_args()


def main():
    args = parse_args()

    checkpoint = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.abspath(os.path.join(SCRIPT_DIR, args.checkpoint))
    output_dir = args.output if os.path.isabs(args.output) else os.path.abspath(os.path.join(SCRIPT_DIR, args.output))

    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        raise FileNotFoundError(f"No images matched: {args.images}")
    if args.max_frames is not None:
        image_paths = image_paths[:args.max_frames]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth_orig_res"), exist_ok=True)
    if args.save_conf:
        os.makedirs(os.path.join(output_dir, "depth_conf"), exist_ok=True)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    print(f"[vggt-single] device={device} dtype={dtype}")
    print(f"[vggt-single] loading checkpoint: {checkpoint}")
    model = _load_model(checkpoint, device)

    manifest = []

    for i, image_path in enumerate(image_paths):
        imgs, coords = load_and_preprocess_images_square([image_path], target_size=518)
        imgs = imgs.to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype, enabled=device.startswith("cuda")):
                pred = model(imgs)

        depth_518 = pred["depth"][0, 0, :, :, 0].float().cpu().numpy().astype(np.float32)
        conf_518 = pred["depth_conf"][0, 0].float().cpu().numpy().astype(np.float32)

        coords_np = coords[0].cpu().numpy().astype(np.float32)
        depth_orig = _crop_depth_to_orig(depth_518, coords_np)

        np.save(os.path.join(output_dir, "depth", f"{i:04d}.npy"), depth_518)
        np.save(os.path.join(output_dir, "depth_orig_res", f"{i:04d}.npy"), depth_orig)
        if args.save_conf:
            np.save(os.path.join(output_dir, "depth_conf", f"{i:04d}.npy"), conf_518)

        manifest.append({
            "index": i,
            "image": image_path,
            "coords_518": coords_np.tolist(),
        })

        if i % 50 == 0:
            print(f"[vggt-single] processed {i+1}/{len(image_paths)}")

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump({"num_frames": len(image_paths), "frames": manifest}, f, indent=2)

    print(f"[vggt-single] done: {output_dir}")


if __name__ == "__main__":
    main()
