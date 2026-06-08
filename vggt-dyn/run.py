"""run.py — CLI entry point for VGGT-Dyn test-time optimization.

Usage:
    python run.py \\
        --images   path/to/frames/*.png \\
        --ckpt     vggt/checkpoints/VGGT-1B.pt \\
        --raft     monst3r/third_party/RAFT/models/raft-things.pth \\
        --output   outputs/my_scene \\
        --niter    50 \\
        --device   cuda

Output (all saved under --output):
    depth/           per-frame refined depth [H, W] as .npy
    extrinsics.npy   [S, 3, 4] cam-from-world
    intrinsics.npy   [S, 3, 3] camera intrinsics
    pts3d.npy        [S, H, W, 3] refined world-frame 3D points
    dynamic_mask/    per-frame dynamic mask [H, W] as .npy (bool)
    metrics.json     final loss values
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ── VGGT repo path ─────────────────────────────────────────────────────────────
_VGGT_REPO = os.path.join(_SCRIPT_DIR, "vggt")
if _VGGT_REPO not in sys.path:
    sys.path.insert(0, _VGGT_REPO)

from vggt_dyn import (
    VGGTInitializer,
    VGGTDynOptimizer,
    optimization_step,
    compute_adjacent_flow,
    get_dynamic_mask_from_optimizer,
)
from vggt_dyn.dynamic_mask import pair_mask_to_frame_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_vggt(ckpt_path: str, device: torch.device):
    """Load VGGT model from local checkpoint."""
    from vggt.models.vggt import VGGT
    model = VGGT()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def _preprocess_images_center_crop(image_paths: list, target_size: int = 518):
    """Preprocess images via center-square crop + resize.

    Unlike letterboxing, this crops the center square from the original image,
    which avoids black-padding for wide/tall images (e.g. KITTI 1216×352).

    Returns:
        imgs_tensor: [S, 3, target_size, target_size] float32 tensor
        original_coords: [S, 6] = [x1, y1, x2, y2, orig_w, orig_h]
            where x1..y2 are in target-pixel space. For center crop,
            the ENTIRE target image corresponds to the cropped region.
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    images = []
    original_coords = []
    to_tensor = TF.to_tensor

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        # Center-square crop
        sq = min(w, h)
        left = (w - sq) // 2
        top  = (h - sq) // 2
        img_crop = img.crop((left, top, left + sq, top + sq))
        # Record original coords in target-px space (full frame after crop)
        # x1,y1=0,0  x2,y2=target,target  and we note the ORIGINAL full frame size
        original_coords.append(np.array([0.0, 0.0, float(target_size), float(target_size),
                                          float(w), float(h)]))
        img_resized = img_crop.resize((target_size, target_size), Image.Resampling.BICUBIC)
        images.append(to_tensor(img_resized))

    imgs_tensor   = torch.stack(images)               # [S, 3, T, T]
    coords_tensor = torch.from_numpy(np.array(original_coords)).float()  # [S, 6]
    return imgs_tensor, coords_tensor


def _preprocess_images_long_edge(image_paths: list, target_long: int = 518):
    """Preprocess images by resizing the long edge to target_long, preserving
    aspect ratio.  Height is rounded to the nearest multiple of 14 (ViT patch
    size requirement).  No padding, no cropping — equivalent to MonST3R's
    crop_img with crop=False.

    For KITTI 1216×352:
        new_width  = 518
        new_height = round(352 * 518/1216 / 14) * 14 = 154
        → tensor shape: [S, 3, 154, 518]

    Returns:
        imgs_tensor:     [S, 3, H_pp, W_pp]  (non-square for most inputs)
        original_coords: [S, 6] = [0, 0, W_pp, H_pp, orig_w, orig_h]
            → _crop_depth_to_orig letterbox path resizes directly to orig
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    images = []
    original_coords = []
    to_tensor = TF.to_tensor

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w >= h:  # wide image (e.g. KITTI)
            new_w = target_long
            new_h = round(h * (new_w / w) / 14) * 14
        else:       # tall image
            new_h = target_long
            new_w = round(w * (new_h / h) / 14) * 14
        new_w = max(new_w, 14)
        new_h = max(new_h, 14)
        img_r = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        original_coords.append(np.array([0.0, 0.0, float(new_w), float(new_h),
                                          float(w), float(h)]))
        images.append(to_tensor(img_r))

    imgs_tensor   = torch.stack(images)                                        # [S, 3, H_pp, W_pp]
    coords_tensor = torch.from_numpy(np.array(original_coords)).float()        # [S, 6]
    return imgs_tensor, coords_tensor


@torch.no_grad()
def run_vggt(model, image_paths: list, device: torch.device,
             preprocess: str = "letterbox") -> dict:
    """Run VGGT feed-forward on a list of image paths.

    preprocess choices:
        'letterbox'   — pad shorter side to square (default, correct for most scenes)
        'center_crop' — crop center square (better for extreme aspect ratios like KITTI)
        'long_edge'   — resize long edge to 518, keep aspect ratio (no black bars,
                        full FOV — closest to MonST3R's preprocessing)

    VGGT forward returns a dict with keys:
        pose_enc [B,S,9], depth [B,S,H,W,1], depth_conf [B,S,H,W],
        world_points [B,S,H,W,3], world_points_conf [B,S,H,W]
    """
    if preprocess == "center_crop":
        imgs_tensor, original_coords = _preprocess_images_center_crop(image_paths, target_size=518)
    elif preprocess == "long_edge":
        imgs_tensor, original_coords = _preprocess_images_long_edge(image_paths, target_long=518)
    else:  # letterbox
        from vggt.utils.load_fn import load_and_preprocess_images_square
        imgs_tensor, original_coords = load_and_preprocess_images_square(image_paths, target_size=518)

    # imgs_tensor: [S, 3, H, W]; add batch dim → [1, S, 3, H, W]
    imgs_tensor = imgs_tensor.unsqueeze(0).to(device)

    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        output = model(imgs_tensor)

    # Store coords so caller can crop depth back to original resolution
    # original_coords: [S, 6] = [x1, y1, x2, y2, orig_w, orig_h] in 518-px space
    output["original_coords"] = original_coords
    output["_preprocess"]     = preprocess
    return output


def load_images_np(paths: list) -> np.ndarray:
    """Load image paths → [S, H_orig, W_orig, 3] uint8 RGB (for RAFT flow)."""
    import cv2
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return np.stack(imgs, axis=0)


def _crop_depth_to_orig(depth_hw: np.ndarray,
                         coords: np.ndarray,
                         mode: str = "letterbox") -> np.ndarray:
    """Convert VGGT-resolution depth back to original image resolution.

    Two modes:
        'letterbox': The 518×518 depth contains black-padded borders.
            coords = [x1, y1, x2, y2, orig_w, orig_h] in 518-px TARGET space.
            → crop ROI [y1:y2, x1:x2] and resize to (orig_h, orig_w).

        'center_crop': The FULL 518×518 depth covers only the center sq×sq
            region of the original image (where sq = min(orig_w, orig_h)).
            coords = [0, 0, 518, 518, orig_w, orig_h] (x1/y1/x2/y2 unused).
            → resize 518×518 → sq×sq, embed at (left, top) in an
              (orig_h, orig_w) zero-canvas.

    Args:
        depth_hw: [H, W] float32 depth at VGGT resolution (e.g. 518×518)
        coords:   [6] array
        mode:     'letterbox' (default) or 'center_crop'

    Returns:
        depth at (orig_h, orig_w); zeros outside crop for center_crop mode.
    """
    import cv2
    x1, y1, x2, y2, orig_w, orig_h = coords
    orig_w, orig_h = int(round(orig_w)), int(round(orig_h))

    if mode == "center_crop":
        sq   = min(orig_w, orig_h)
        left = (orig_w - sq) // 2
        top  = (orig_h - sq) // 2
        # Resize 518×518 → sq×sq (original crop resolution)
        depth_sq = cv2.resize(depth_hw, (sq, sq), interpolation=cv2.INTER_LINEAR)
        # Embed in full-size zero canvas
        canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
        canvas[top:top + sq, left:left + sq] = depth_sq
        return canvas
    else:  # letterbox
        r1, r2 = int(round(y1)), int(round(y2))
        c1, c2 = int(round(x1)), int(round(x2))
        H, W = depth_hw.shape
        r1, r2 = max(0, r1), min(H, r2)
        c1, c2 = max(0, c1), min(W, c2)
        crop = depth_hw[r1:r2, c1:c2]
        if crop.size == 0:
            crop = depth_hw
        out = cv2.resize(crop, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)


def save_outputs(output_dir: str, pts3d, depth, extrinsics, intrinsics,
                 dynamic_mask_per_frame, metrics: dict,
                 original_coords=None, preprocess: str = "letterbox"):
    """Save all refined outputs to disk."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "dynamic_mask"), exist_ok=True)

    np.save(os.path.join(output_dir, "pts3d.npy"),
            pts3d.cpu().numpy())
    np.save(os.path.join(output_dir, "extrinsics.npy"),
            extrinsics.cpu().numpy())
    np.save(os.path.join(output_dir, "intrinsics.npy"),
            intrinsics.cpu().numpy())

    depth_np = depth.cpu().numpy()           # [S, H, W]
    mask_np  = dynamic_mask_per_frame.cpu().numpy()  # [S, 1, H, W]

    # If original_coords are provided, also save depth at original resolution
    # under depth_orig_res/ so evaluators can use the right spatial layout.
    coords_np = None
    if original_coords is not None:
        coords_np = original_coords.cpu().numpy() if hasattr(original_coords, "cpu") \
                    else np.array(original_coords)
        os.makedirs(os.path.join(output_dir, "depth_orig_res"), exist_ok=True)
        np.save(os.path.join(output_dir, "preprocess_coords.npy"), coords_np)

    S = depth_np.shape[0]
    for i in range(S):
        np.save(os.path.join(output_dir, "depth",        f"{i:04d}.npy"), depth_np[i])
        np.save(os.path.join(output_dir, "dynamic_mask", f"{i:04d}.npy"), mask_np[i, 0])
        if coords_np is not None:
            d_orig = _crop_depth_to_orig(depth_np[i], coords_np[i], mode=preprocess)
            np.save(os.path.join(output_dir, "depth_orig_res", f"{i:04d}.npy"), d_orig)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[vggt-dyn] results saved → {output_dir}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[vggt-dyn] device = {device}")

    # ── 1. Load images ────────────────────────────────────────────────────────
    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        raise FileNotFoundError(f"No images found matching: {args.images}")
    if args.max_frames is not None and len(image_paths) > args.max_frames:
        print(f"[vggt-dyn] capping frames {len(image_paths)} → {args.max_frames} (--max_frames)")
        image_paths = image_paths[:args.max_frames]
    S = len(image_paths)
    print(f"[vggt-dyn] {S} images loaded")

    # ── 2. VGGT feed-forward ──────────────────────────────────────────────────
    print(f"[vggt-dyn] running VGGT (preprocess={args.preprocess}) ...")
    vggt_model = load_vggt(args.ckpt, device)
    vggt_out   = run_vggt(vggt_model, image_paths, device, preprocess=args.preprocess)
    del vggt_model
    torch.cuda.empty_cache()

    H_vggt = vggt_out["depth"].shape[2]
    W_vggt = vggt_out["depth"].shape[3]
    original_coords = vggt_out.pop("original_coords", None)  # [S,6] coords
    _preprocess     = vggt_out.pop("_preprocess", "letterbox")
    print(f"[vggt-dyn] VGGT output: S={S}, H={H_vggt}, W={W_vggt}")

    # ── 3. RAFT optical flow (adjacent pairs only) ─────────────────────────────────────
    # CRITICAL: RAFT must see the SAME preprocessed 518×518 images that VGGT used.
    # Using squeezed/distorted images produces flow in a different coordinate system
    # than VGGT's ego-flow, making the residual comparison meaningless.
    print("[vggt-dyn] computing RAFT optical flow ...")
    # Re-preprocess using the same mode chosen for VGGT
    if args.preprocess == "center_crop":
        imgs_tensor_pp, _ = _preprocess_images_center_crop(image_paths, target_size=H_vggt)
    elif args.preprocess == "long_edge":
        imgs_tensor_pp, _ = _preprocess_images_long_edge(image_paths, target_long=max(H_vggt, W_vggt))
    else:
        from vggt.utils.load_fn import load_and_preprocess_images_square
        imgs_tensor_pp, _ = load_and_preprocess_images_square(image_paths, target_size=H_vggt)
    imgs_norm = imgs_tensor_pp.permute(0, 2, 3, 1).numpy().astype(np.float32)
    # imgs_norm: [S, H_vggt, W_vggt, 3] — same pixel space as VGGT depth/pose

    flow_fwd, flow_bwd, valid_fwd, valid_bwd = compute_adjacent_flow(
        imgs_norm, model_path=args.raft, device=str(device)
    )
    del flow_bwd, valid_bwd

    # ── 4. Initialise optimizer ───────────────────────────────────────────────
    print("[vggt-dyn] initialising optimizer ...")
    init = VGGTInitializer(vggt_out, (H_vggt, W_vggt))
    init.to(device)

    flow_fwd  = flow_fwd.to(device)
    valid_fwd = valid_fwd.to(device)

    net  = VGGTDynOptimizer(
        init,
        flow_fwd,
        valid_fwd,
        anchor_weight       = args.anchor_weight,
        flow_weight         = args.flow_weight,
        depth_reg_weight    = args.depth_reg_weight,
        mon_smooth_weight   = args.mon_smooth_weight,
        dyn_pointmap_weight = args.dyn_pointmap_weight,
        track_smooth_weight = args.track_smooth_weight,
        loss_version        = args.loss_version,
    ).to(device)

    adam = torch.optim.Adam(net.parameters(), lr=args.lr)

    # ── 5. Optimization loop ──────────────────────────────────────────────────
    print(f"[vggt-dyn] optimising for {args.niter} iterations (loss={args.loss_version}) ...")
    loss_history = []

    for it in range(args.niter):
        total_loss, flow_loss, lr = optimization_step(
            net, it, args.niter, args.lr, args.lr_min, adam
        )
        loss_history.append(float(total_loss))

        if it % args.mask_refresh == (args.mask_refresh - 1):
            mask = get_dynamic_mask_from_optimizer(
                net,
                threshold=args.mask_threshold,
                normalize=True,
            )
            net.update_dynamic_mask(mask)

        if args.verbose and it % 10 == 0:
            sec_name = "flow_loss" if args.loss_version == "mon" else "dyn_loss"
            print(f"  iter {it:3d}/{args.niter}  loss={total_loss:.4f}  "
                  f"{sec_name}={flow_loss:.4f}  lr={lr:.2e}")

    print(f"[vggt-dyn] final loss: {loss_history[-1]:.4f} "
          f"(was {loss_history[0]:.4f})")

    # ── 6. Extract and save results ───────────────────────────────────────────
    print("[vggt-dyn] saving results ...")
    pts3d      = net.get_pts3d()        # [S, H, W, 3]
    depth      = net.get_depth()        # [S, H, W]
    extrinsics = net.get_extrinsics()   # [S, 3, 4]
    intrinsics = net.get_K()            # [S, 3, 3]

    # Convert per-pair dynamic mask → per-frame [S, 1, H, W]
    pair_mask  = net.dynamic_mask       # [S-1, 1, H, W]
    frame_mask = pair_mask_to_frame_mask(pair_mask, S)  # [S, 1, H, W]

    metrics = {
        "niter":      args.niter,
        "loss_init":  loss_history[0],
        "loss_final": loss_history[-1],
        "loss_history": loss_history,
        "preprocess": _preprocess,
    }

    save_outputs(
        args.output,
        pts3d, depth, extrinsics, intrinsics,
        frame_mask, metrics,
        original_coords=original_coords,
        preprocess=_preprocess,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VGGT-Dyn: test-time optimization for dynamic scenes")

    # I/O
    p.add_argument("--images",      required=True,  help="glob pattern for input frames (e.g. 'frames/*.png')")
    p.add_argument("--ckpt",        required=True,  help="VGGT checkpoint (.pt)")
    p.add_argument("--raft",        required=True,  help="RAFT model weights (.pth)")
    p.add_argument("--output",      default="outputs/vggt_dyn", help="output directory")
    p.add_argument("--max_frames",  type=int, default=None,
                   help="cap number of frames (first N); useful to avoid OOM on long sequences")
    p.add_argument("--preprocess", default="letterbox",
                   choices=["letterbox", "center_crop", "long_edge"],
                   help="image preprocessing: "
                        "'letterbox' (pad to square, default), "
                        "'center_crop' (crop center square), "
                        "'long_edge' (resize long edge to 518, keep aspect ratio — closest to MonST3R)")

    # Optimization
    p.add_argument("--niter",        type=int,   default=50,    help="number of optimization iterations")
    p.add_argument("--lr",           type=float, default=1e-3,  help="peak learning rate (cosine schedule)")
    p.add_argument("--lr_min",       type=float, default=1e-5,  help="minimum learning rate")
    p.add_argument("--loss_version", default="mon", choices=["mon", "dyn"],
                   help="mon: anchor+flow+depth_reg (original)  "
                        "dyn: static_rigid+dyn_pointmap+track_smooth")
    # mon weights
    p.add_argument("--anchor_weight",     type=float, default=1.0,  help="[mon] anchor loss weight")
    p.add_argument("--flow_weight",       type=float, default=1.0,  help="[mon] flow loss weight")
    p.add_argument("--depth_reg_weight",  type=float, default=0.1,  help="[mon] depth regularization weight")
    p.add_argument("--mon_smooth_weight", type=float, default=0.1,  help="[mon] global 3D track smoothness weight")
    # dyn weights
    p.add_argument("--dyn_pointmap_weight", type=float, default=1.0,
                   help="[dyn] λ — dynamic pointmap regression weight")
    p.add_argument("--track_smooth_weight", type=float, default=0.1,
                   help="[dyn] γ — 3D track smoothness weight")

    # Dynamic mask
    p.add_argument("--mask_refresh",   type=int,   default=10,   help="refresh dynamic mask every N iters")
    p.add_argument("--mask_threshold", type=float, default=0.35, help="normalized flow-residual threshold")

    # Misc
    p.add_argument("--device",  default="cuda")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
