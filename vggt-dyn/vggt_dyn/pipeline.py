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
from vggt_dyn.losses import MonLoss, DynLoss
from vggt_dyn.optimizer import VGGTDynOptimizer, optimization_step
from vggt_dyn.dynamic_mask import get_dynamic_mask_from_optimizer, pair_mask_to_frame_mask, frame_mask_to_pair_mask
from vggt_dyn.utils.flow_utils import compute_adjacent_flow


# ===========================================================================
# Model loading
# ===========================================================================

def load_vggt(ckpt_path: str, device: torch.device):
    """Load VGGT model from a local checkpoint.

    Accepts both raw state-dicts (original VGGT release) and full finetune
    checkpoints saved by finetune/model_utils.py::save_checkpoint, which
    wrap the state-dict under the 'model' key alongside optimizer/scaler/args.
    """
    from vggt.models.vggt import VGGT
    model = VGGT()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
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

def _load_gt_depth(
    gt_depth_dir: str,
    image_paths: list,
    H: int,
    W: int,
    device: torch.device,
    preprocess: str = "center_crop",
) -> torch.Tensor:
    """Load Sintel GT depth (.dpt) files, crop+resize to (H, W), return [S, H, W]."""
    import cv2

    SINTEL_TAG = 202021.25

    def _read_dpt(path: str) -> np.ndarray:
        with open(path, "rb") as f:
            tag = np.frombuffer(f.read(4), dtype=np.float32)[0]
            if abs(float(tag) - SINTEL_TAG) > 1e-4:
                raise ValueError(f"Bad .dpt tag in {path}: {tag}")
            width  = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            height = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            data   = np.frombuffer(f.read(width * height * 4), dtype=np.float32)
        d = data.reshape(height, width).astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        d[d < 0] = 0.0
        return d

    dpt_files = sorted(
        os.path.join(gt_depth_dir, f)
        for f in os.listdir(gt_depth_dir) if f.endswith(".dpt")
    )[:len(image_paths)]

    if not dpt_files:
        raise FileNotFoundError(f"No .dpt files found in {gt_depth_dir}")

    depths = []
    orig_h = orig_w = None
    for dpt_path in dpt_files:
        d_full = _read_dpt(dpt_path)
        if orig_h is None:
            orig_h, orig_w = d_full.shape
        if preprocess == "center_crop":
            sq   = min(orig_w, orig_h)
            left = (orig_w - sq) // 2
            top  = (orig_h - sq) // 2
            d_crop = d_full[top:top + sq, left:left + sq]
            d_out  = cv2.resize(d_crop, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            d_out = cv2.resize(d_full, (W, H), interpolation=cv2.INTER_LINEAR)
        depths.append(d_out.astype(np.float32))

    return torch.from_numpy(np.stack(depths)).float().to(device)


def _load_gt_pair_mask(
    gt_mask_dir: str,
    image_paths: list,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Load GT dynamic-mask PNGs and convert to per-pair bool tensor [S-1,1,H,W].

    PNG basenames must match those in image_paths (e.g. frame_0001.png).
    Missing files are treated as all-static (zeros).
    """
    from PIL import Image

    S = len(image_paths)
    frame_masks = []
    for p in image_paths:
        mask_path = os.path.join(gt_mask_dir, os.path.basename(p))
        if os.path.isfile(mask_path):
            m = Image.open(mask_path).convert("L").resize((W, H), Image.Resampling.NEAREST)
            m_t = torch.from_numpy(np.array(m, dtype=np.float32) / 255.0) > 0.5  # [H, W]
        else:
            m_t = torch.zeros(H, W, dtype=torch.bool)
        frame_masks.append(m_t.unsqueeze(0))  # [1, H, W]

    frame_mask = torch.stack(frame_masks, dim=0).to(device)  # [S, 1, H, W]
    return frame_mask_to_pair_mask(frame_mask)               # [S-1, 1, H, W]


def save_outputs(output_dir: str, pts3d, depth, extrinsics, intrinsics,
                 dynamic_mask_per_frame, metrics: dict,
                 original_coords=None, preprocess: str = "letterbox",
                 depth_conf=None):
    """Save all refined outputs to disk."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "dynamic_mask"), exist_ok=True)
    if depth_conf is not None:
        os.makedirs(os.path.join(output_dir, "depth_conf"), exist_ok=True)

    np.save(os.path.join(output_dir, "pts3d.npy"),       pts3d.cpu().numpy())
    np.save(os.path.join(output_dir, "extrinsics.npy"),  extrinsics.cpu().numpy())
    np.save(os.path.join(output_dir, "intrinsics.npy"),  intrinsics.cpu().numpy())

    depth_np = depth.cpu().numpy()
    mask_np  = dynamic_mask_per_frame.cpu().numpy()
    conf_np  = depth_conf.cpu().numpy() if depth_conf is not None else None

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
        if conf_np is not None:
            np.save(os.path.join(output_dir, "depth_conf", f"{i:04d}.npy"), conf_np[i])
        if coords_np is not None:
            d_orig = _crop_depth_to_orig(depth_np[i], coords_np[i], mode=preprocess)
            np.save(os.path.join(output_dir, "depth_orig_res", f"{i:04d}.npy"), d_orig)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[vggt-dyn] results saved → {output_dir}")


def _save_niter0_outputs(args, init, image_paths, original_coords, preprocess):
    """Save raw VGGT init state when niter=0 (no TTO, no RAFT)."""
    S = init.S
    init_ext = torch.cat([init.R, init.T.unsqueeze(-1)], dim=-1)  # [S, 3, 4]
    frame_mask = torch.zeros(S, 1, init.H, init.W, dtype=torch.bool, device=init.R.device)

    conf_np = init.anchor_conf.detach().cpu().numpy()
    conf_stats = {
        "min": float(conf_np.min()), "max": float(conf_np.max()),
        "mean": float(conf_np.mean()), "std": float(conf_np.std()),
    }
    metrics = {
        "niter": 0,
        "loss_init": None, "loss_final": None, "loss_history": [],
        "preprocess": preprocess,
        "anchor_conf_stats": conf_stats,
        "freeze_pose": getattr(args, "freeze_pose", False),
        "images_glob": args.images,
        "image_paths": [os.path.relpath(p, args.output) for p in image_paths],
    }
    save_outputs(
        args.output,
        init.anchor_pts, init.depth, init_ext, init.K,
        frame_mask, metrics,
        original_coords=original_coords, preprocess=preprocess,
        depth_conf=init.depth_conf,
    )
    np.save(os.path.join(args.output, "init_extrinsics.npy"), init_ext.cpu().numpy())


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

    init_dir = getattr(args, "init_dir", None)
    if not init_dir and not getattr(args, "ckpt", None):
        raise ValueError("Either --ckpt or --init_dir must be provided")
    if args.niter > 0 and not getattr(args, "raft", None):
        raise ValueError("--raft is required when niter > 0")

    # ── 1. Load images ────────────────────────────────────────────────────────
    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        raise FileNotFoundError(f"No images found matching: {args.images}")
    if args.max_frames is not None and len(image_paths) > args.max_frames:
        print(f"[vggt-dyn] capping frames {len(image_paths)} → {args.max_frames}")
        image_paths = image_paths[:args.max_frames]
    S = len(image_paths)
    print(f"[vggt-dyn] {S} images loaded")

    # ── 2. VGGT feed-forward (or load from --init_dir) ───────────────────────
    init_dir = getattr(args, "init_dir", None)
    if init_dir:
        print(f"[vggt-dyn] loading init state from {init_dir} ...")
        init = VGGTInitializer.from_dir(init_dir, device)
        H_vggt, W_vggt = init.H, init.W
        original_coords = None
        _preprocess = args.preprocess
        print(f"[vggt-dyn] loaded: S={init.S}, H={H_vggt}, W={W_vggt}")
    else:
        vggt_model  = load_vggt(args.ckpt, device)
        window_size = getattr(args, "window_size", 0)
        _preprocess = args.preprocess

        if window_size > 0:
            # ── windowed VGGT (Scheme B): multi-window forward + pose graph ──
            print(f"[vggt-dyn] windowed VGGT (W={window_size}, "
                  f"stride={getattr(args, 'window_stride', 5)}, "
                  f"preprocess={_preprocess}) ...")
            from vggt_dyn.windowed_vggt import run_windowed_vggt
            merged_state = run_windowed_vggt(
                vggt_model, image_paths, device,
                preprocess   = _preprocess,
                window_size  = window_size,
                stride       = getattr(args, "window_stride", 5),
            )
            del vggt_model
            torch.cuda.empty_cache()

            H_vggt = merged_state["depth"].shape[1]
            W_vggt = merged_state["depth"].shape[2]
            original_coords = None
            print(f"[vggt-dyn] merged: S={S}, H={H_vggt}, W={W_vggt}")

            init = VGGTInitializer.__new__(VGGTInitializer)
            init.image_size_hw = (H_vggt, W_vggt)
            init._state = merged_state

        else:
            # ── single-pass VGGT (original path) ────────────────────────────
            print(f"[vggt-dyn] running VGGT (preprocess={_preprocess}) ...")
            vggt_out = run_vggt(vggt_model, image_paths, device, preprocess=_preprocess)
            del vggt_model
            torch.cuda.empty_cache()

            H_vggt = vggt_out["depth"].shape[2]
            W_vggt = vggt_out["depth"].shape[3]
            original_coords = vggt_out.pop("original_coords", None)
            _preprocess     = vggt_out.pop("_preprocess", "letterbox")
            print(f"[vggt-dyn] VGGT output: S={S}, H={H_vggt}, W={W_vggt}")

            init = VGGTInitializer(vggt_out, (H_vggt, W_vggt))

        init.to(device)

    # niter=0: skip RAFT and TTO, save raw init state and return
    if args.niter == 0:
        print("[vggt-dyn] niter=0, skipping RAFT and TTO ...")
        _save_niter0_outputs(args, init, image_paths, original_coords, _preprocess)
        return

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
    flow_fwd  = flow_fwd.to(device)
    valid_fwd = valid_fwd.to(device)

    # ── GT depth injection (replaces VGGT depth at init, pose unchanged) ──────
    gt_depth_dir = getattr(args, "gt_depth_dir", None)
    if gt_depth_dir:
        print(f"[vggt-dyn] injecting GT depth from {gt_depth_dir} ...")
        gt_depth = _load_gt_depth(gt_depth_dir, image_paths, H_vggt, W_vggt, device, args.preprocess)
        init._state["depth"] = gt_depth
        print(f"[vggt-dyn] GT depth injected: median={gt_depth[gt_depth > 0].median():.3f}")

    if args.loss_version == "dyn":
        loss_fn = DynLoss(
            dyn_pointmap_weight = args.dyn_pointmap_weight,
            track_smooth_weight = args.track_smooth_weight,
        )
    else:
        loss_fn = MonLoss(
            anchor_weight        = args.anchor_weight,
            flow_weight          = args.flow_weight,
            depth_reg_weight     = args.depth_reg_weight,
            smooth_weight        = args.smooth_weight,
            scale_flow_loss      = getattr(args, "scale_flow_loss", False),
            translation_weight   = getattr(args, "translation_weight", 0.1),
            niter                = args.niter,
            flow_loss_start_frac = getattr(args, "flow_loss_start_frac", 0.15),
            flow_loss_thre       = getattr(args, "flow_loss_thre", 50.0),
            pxl_thre             = getattr(args, "pxl_thre", 50.0),
        )

    net = VGGTDynOptimizer(
        init, flow_fwd, valid_fwd,
        loss        = loss_fn,
        freeze_pose = args.freeze_pose,
    ).to(device)

    # ── pre-optimization (raw VGGT) extrinsics & conf stats, for diagnostics ──
    init_extrinsics = torch.cat([init.R, init.T.unsqueeze(-1)], dim=-1)  # [S, 3, 4]
    conf_np  = init.anchor_conf.detach().cpu().numpy()
    conf_hist, conf_bin_edges = np.histogram(conf_np, bins=10)
    conf_stats = {
        "min":  float(conf_np.min()),
        "max":  float(conf_np.max()),
        "mean": float(conf_np.mean()),
        "std":  float(conf_np.std()),
        "histogram":  conf_hist.tolist(),
        "bin_edges":  conf_bin_edges.tolist(),
    }

    adam = torch.optim.Adam(net.parameters(), lr=args.lr)

    # ── GT mask injection (skips self-mask computation if provided) ───────────
    gt_mask_dir = getattr(args, "gt_mask_dir", None)
    if gt_mask_dir:
        print(f"[vggt-dyn] loading GT dynamic mask from {gt_mask_dir} ...")
        gt_pair_mask = _load_gt_pair_mask(gt_mask_dir, image_paths, H_vggt, W_vggt, device)
        net.update_dynamic_mask(gt_pair_mask)
        _mask_refresh = args.niter + 1  # never recompute self-mask during TTO
        print(f"[vggt-dyn] GT mask loaded — self-mask recomputation disabled")
    else:
        _mask_refresh = args.mask_refresh

    # ── 5. Optimization loop ──────────────────────────────────────────────────
    print(f"[vggt-dyn] optimising {args.niter} iters (loss={args.loss_version}) ...")
    loss_history = []
    eval_every = getattr(args, "eval_every", 0)
    if eval_every > 0:
        os.makedirs(os.path.join(args.output, "checkpoints"), exist_ok=True)
        # iter_0000: pre-optimization (raw VGGT) state, for before/after comparison
        ckpt_dir = os.path.join(args.output, "checkpoints", "iter_0000")
        os.makedirs(ckpt_dir, exist_ok=True)
        with torch.no_grad():
            np.save(os.path.join(ckpt_dir, "extrinsics.npy"), net.get_extrinsics().cpu().numpy())
            np.save(os.path.join(ckpt_dir, "intrinsics.npy"), net.get_K().cpu().numpy())
            depth_ckpt = net.get_depth().cpu().numpy()
            os.makedirs(os.path.join(ckpt_dir, "depth"), exist_ok=True)
            for i in range(depth_ckpt.shape[0]):
                np.save(os.path.join(ckpt_dir, "depth", f"{i:04d}.npy"), depth_ckpt[i])
            frame_mask_ckpt = pair_mask_to_frame_mask(net.dynamic_mask, S).cpu().numpy()
            os.makedirs(os.path.join(ckpt_dir, "dynamic_mask"), exist_ok=True)
            for i in range(frame_mask_ckpt.shape[0]):
                np.save(os.path.join(ckpt_dir, "dynamic_mask", f"{i:04d}.npy"), frame_mask_ckpt[i])
        with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
            json.dump({"iter": 0, "loss": None}, f, indent=2)

    for it in range(args.niter):
        total_loss, flow_loss, lr = optimization_step(
            net, it, args.niter, args.lr, args.lr_min, adam
        )
        loss_history.append(float(total_loss))

        if it % _mask_refresh == (_mask_refresh - 1):
            mask = get_dynamic_mask_from_optimizer(
                net, threshold=args.mask_threshold, normalize=True,
            )
            net.update_dynamic_mask(mask)

        if args.verbose and it % 10 == 0:
            sec_name = "flow_loss" if args.loss_version == "mon" else "dyn_loss"
            print(f"  iter {it:3d}/{args.niter}  loss={total_loss:.4f}  "
                  f"{sec_name}={flow_loss:.4f}  lr={lr:.2e}")

        if eval_every > 0 and ((it + 1) % eval_every == 0 or it == args.niter - 1):
            ckpt_dir = os.path.join(args.output, "checkpoints", f"iter_{it+1:04d}")
            os.makedirs(ckpt_dir, exist_ok=True)
            with torch.no_grad():
                np.save(os.path.join(ckpt_dir, "extrinsics.npy"), net.get_extrinsics().cpu().numpy())
                np.save(os.path.join(ckpt_dir, "intrinsics.npy"), net.get_K().cpu().numpy())
                depth_ckpt = net.get_depth().cpu().numpy()
                os.makedirs(os.path.join(ckpt_dir, "depth"), exist_ok=True)
                for i in range(depth_ckpt.shape[0]):
                    np.save(os.path.join(ckpt_dir, "depth", f"{i:04d}.npy"), depth_ckpt[i])
                frame_mask_ckpt = pair_mask_to_frame_mask(net.dynamic_mask, S).cpu().numpy()
                os.makedirs(os.path.join(ckpt_dir, "dynamic_mask"), exist_ok=True)
                for i in range(frame_mask_ckpt.shape[0]):
                    np.save(os.path.join(ckpt_dir, "dynamic_mask", f"{i:04d}.npy"), frame_mask_ckpt[i])
            with open(os.path.join(ckpt_dir, "metrics.json"), "w") as f:
                json.dump({"iter": it + 1, "loss": float(total_loss)}, f, indent=2)

    if loss_history:
        print(f"[vggt-dyn] final loss: {loss_history[-1]:.4f} (was {loss_history[0]:.4f})")
    else:
        print("[vggt-dyn] niter=0, no optimization")

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
        "loss_init":    loss_history[0] if loss_history else None,
        "loss_final":   loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "preprocess":   _preprocess,
        "anchor_conf_stats": conf_stats,
        "freeze_pose":  args.freeze_pose,
        "images_glob":  args.images,
        "image_paths":  [os.path.relpath(p, args.output) for p in image_paths],
    }

    save_outputs(
        args.output,
        pts3d, depth, extrinsics, intrinsics, frame_mask, metrics,
        original_coords=original_coords, preprocess=_preprocess,
        depth_conf=init.depth_conf,
    )
    np.save(os.path.join(args.output, "init_extrinsics.npy"), init_extrinsics.cpu().numpy())
