"""pipeline.py — Core VGGT-Dyn test-time optimization pipeline.

This module extracts the pipeline logic from run.py so it can be imported
and used programmatically, without going through the CLI.

Public API
----------
    from vggt_dyn.pipeline import run_pipeline, load_vggt, run_vggt, save_outputs
"""

import os
import sys
import json
import glob
import numpy as np
import torch

# ── ensure VGGT repo is importable when this module is loaded directly ────────
_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))   # vggt_dyn/
_PROJECT_DIR = os.path.dirname(_MODULE_DIR)                  # vggt-dyn/
_VGGT_REPO   = os.path.join(_PROJECT_DIR, "vggt")
for _p in (_PROJECT_DIR, _VGGT_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vggt_dyn.initializer import VGGTInitializer
from vggt_dyn.optimizer import VGGTDynOptimizer, optimization_step
from vggt_dyn.dynamic_mask import get_dynamic_mask_from_optimizer, pair_mask_to_frame_mask
from vggt_dyn.utils.flow_utils import compute_adjacent_flow


# ===========================================================================
# Model loading
# ===========================================================================

def load_vggt(ckpt_path: str, device: torch.device):
    """Load VGGT model from a local checkpoint."""
    from vggt.models.vggt import VGGT
    model = VGGT()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


# ===========================================================================
# Image preprocessing
# ===========================================================================

def _preprocess_images_center_crop(image_paths: list, target_size: int = 518):
    """Center-square crop + resize.

    Returns:
        imgs_tensor:     [S, 3, target_size, target_size]
        original_coords: [S, 6] = [x1, y1, x2, y2, orig_w, orig_h]
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    images = []
    original_coords = []
    to_tensor = TF.to_tensor

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        sq   = min(w, h)
        left = (w - sq) // 2
        top  = (h - sq) // 2
        img_crop = img.crop((left, top, left + sq, top + sq))
        original_coords.append(
            np.array([0.0, 0.0, float(target_size), float(target_size),
                      float(w), float(h)])
        )
        images.append(to_tensor(img_crop.resize((target_size, target_size),
                                                Image.Resampling.BICUBIC)))

    imgs_tensor   = torch.stack(images)
    coords_tensor = torch.from_numpy(np.array(original_coords)).float()
    return imgs_tensor, coords_tensor


def _preprocess_images_long_edge(image_paths: list, target_long: int = 518):
    """Resize long edge to target_long, keep aspect ratio.

    Height rounded to nearest multiple of 14 (ViT patch size).
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    images = []
    original_coords = []
    to_tensor = TF.to_tensor

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w >= h:
            new_w = target_long
            new_h = round(h * (new_w / w) / 14) * 14
        else:
            new_h = target_long
            new_w = round(w * (new_h / h) / 14) * 14
        new_w = max(new_w, 14)
        new_h = max(new_h, 14)
        img_r = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        original_coords.append(
            np.array([0.0, 0.0, float(new_w), float(new_h), float(w), float(h)])
        )
        images.append(to_tensor(img_r))

    imgs_tensor   = torch.stack(images)
    coords_tensor = torch.from_numpy(np.array(original_coords)).float()
    return imgs_tensor, coords_tensor


@torch.no_grad()
def run_vggt(model, image_paths: list, device: torch.device,
             preprocess: str = "letterbox") -> dict:
    """Run VGGT feed-forward.

    preprocess choices:
        'letterbox'   — pad shorter side to square
        'center_crop' — crop center square (better for extreme aspect ratios)
        'long_edge'   — resize long edge to 518, keep aspect ratio
    """
    if preprocess == "center_crop":
        imgs_tensor, original_coords = _preprocess_images_center_crop(image_paths, 518)
    elif preprocess == "long_edge":
        imgs_tensor, original_coords = _preprocess_images_long_edge(image_paths, 518)
    else:
        from vggt.utils.load_fn import load_and_preprocess_images_square
        imgs_tensor, original_coords = load_and_preprocess_images_square(
            image_paths, target_size=518)

    imgs_tensor = imgs_tensor.unsqueeze(0).to(device)
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        output = model(imgs_tensor)

    output["original_coords"] = original_coords
    output["_preprocess"]     = preprocess
    return output


def load_images_np(paths: list) -> np.ndarray:
    """Load image paths → [S, H_orig, W_orig, 3] uint8 RGB."""
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
    """Convert VGGT-resolution depth back to original image resolution."""
    import cv2
    x1, y1, x2, y2, orig_w, orig_h = coords
    orig_w, orig_h = int(round(orig_w)), int(round(orig_h))

    if mode == "center_crop":
        sq   = min(orig_w, orig_h)
        left = (orig_w - sq) // 2
        top  = (orig_h - sq) // 2
        depth_sq = cv2.resize(depth_hw, (sq, sq), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
        canvas[top:top + sq, left:left + sq] = depth_sq
        return canvas
    else:
        r1, r2 = int(round(y1)), int(round(y2))
        c1, c2 = int(round(x1)), int(round(x2))
        H, W = depth_hw.shape
        r1, r2 = max(0, r1), min(H, r2)
        c1, c2 = max(0, c1), min(W, c2)
        crop = depth_hw[r1:r2, c1:c2]
        if crop.size == 0:
            crop = depth_hw
        return cv2.resize(crop, (orig_w, orig_h),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)


# ===========================================================================
# Output saving
# ===========================================================================

def save_outputs(output_dir: str, pts3d, depth, extrinsics, intrinsics,
                 dynamic_mask_per_frame, metrics: dict,
                 original_coords=None, preprocess: str = "letterbox"):
    """Save all refined outputs to disk."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "dynamic_mask"), exist_ok=True)

    np.save(os.path.join(output_dir, "pts3d.npy"),       pts3d.cpu().numpy())
    np.save(os.path.join(output_dir, "extrinsics.npy"),  extrinsics.cpu().numpy())
    np.save(os.path.join(output_dir, "intrinsics.npy"),  intrinsics.cpu().numpy())

    depth_np = depth.cpu().numpy()
    mask_np  = dynamic_mask_per_frame.cpu().numpy()

    coords_np = None
    if original_coords is not None:
        coords_np = original_coords.cpu().numpy() \
            if hasattr(original_coords, "cpu") else np.array(original_coords)
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


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(args):
    """Full VGGT-Dyn test-time optimization pipeline.

    Args come from run.py's parse_args(), but any object with the same
    attributes works (useful for programmatic use).
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[vggt-dyn] device = {device}")

    # ── 1. Load images ────────────────────────────────────────────────────────
    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        raise FileNotFoundError(f"No images found matching: {args.images}")
    if args.max_frames is not None and len(image_paths) > args.max_frames:
        print(f"[vggt-dyn] capping frames {len(image_paths)} → {args.max_frames}")
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
    original_coords = vggt_out.pop("original_coords", None)
    _preprocess     = vggt_out.pop("_preprocess", "letterbox")
    print(f"[vggt-dyn] VGGT output: S={S}, H={H_vggt}, W={W_vggt}")

    # ── 3. RAFT optical flow ──────────────────────────────────────────────────
    print("[vggt-dyn] computing RAFT optical flow ...")
    if args.preprocess == "center_crop":
        imgs_tensor_pp, _ = _preprocess_images_center_crop(image_paths, target_size=H_vggt)
    elif args.preprocess == "long_edge":
        imgs_tensor_pp, _ = _preprocess_images_long_edge(
            image_paths, target_long=max(H_vggt, W_vggt))
    else:
        from vggt.utils.load_fn import load_and_preprocess_images_square
        imgs_tensor_pp, _ = load_and_preprocess_images_square(
            image_paths, target_size=H_vggt)
    imgs_norm = imgs_tensor_pp.permute(0, 2, 3, 1).numpy().astype(np.float32)

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

    net = VGGTDynOptimizer(
        init, flow_fwd, valid_fwd,
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
    print(f"[vggt-dyn] optimising {args.niter} iters (loss={args.loss_version}) ...")
    loss_history = []

    for it in range(args.niter):
        total_loss, flow_loss, lr = optimization_step(
            net, it, args.niter, args.lr, args.lr_min, adam
        )
        loss_history.append(float(total_loss))

        if it % args.mask_refresh == (args.mask_refresh - 1):
            mask = get_dynamic_mask_from_optimizer(
                net, threshold=args.mask_threshold, normalize=True,
            )
            net.update_dynamic_mask(mask)

        if args.verbose and it % 10 == 0:
            sec_name = "flow_loss" if args.loss_version == "mon" else "dyn_loss"
            print(f"  iter {it:3d}/{args.niter}  loss={total_loss:.4f}  "
                  f"{sec_name}={flow_loss:.4f}  lr={lr:.2e}")

    print(f"[vggt-dyn] final loss: {loss_history[-1]:.4f} (was {loss_history[0]:.4f})")

    # ── 6. Extract and save results ───────────────────────────────────────────
    print("[vggt-dyn] saving results ...")
    pts3d      = net.get_pts3d()
    depth      = net.get_depth()
    extrinsics = net.get_extrinsics()
    intrinsics = net.get_K()
    pair_mask  = net.dynamic_mask
    frame_mask = pair_mask_to_frame_mask(pair_mask, S)

    metrics = {
        "niter":        args.niter,
        "loss_init":    loss_history[0],
        "loss_final":   loss_history[-1],
        "loss_history": loss_history,
        "preprocess":   _preprocess,
    }

    save_outputs(
        args.output,
        pts3d, depth, extrinsics, intrinsics, frame_mask, metrics,
        original_coords=original_coords, preprocess=_preprocess,
    )
