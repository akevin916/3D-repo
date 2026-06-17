#!/usr/bin/env python3
"""visualize.py — Diagnostic visualization for vggt-dyn run.py outputs.

Subcommands
───────────
  gif      Quick GIF: RGB / Depth / Dynamic-mask  (+ optional RAFT / Ego-flow)
           Output: <output_dir>/viz.gif, <output_dir>/stats.json

  inspect  Detailed 6-panel per-frame diagnostic with ego-flow, residual,
           threshold sweep, optional GT depth Abs-Rel eval
           Output: <save_dir>/panels/XXXX.png, <save_dir>/stats.json

Usage
─────
  python visualize.py gif \
      --output_dir outputs/alley_2 [--raft checkpoints/raft/raft-things.pth]

  python visualize.py inspect \
      --images "data/sintel/training/final/alley_2/*.png" \
      --from_run outputs/alley_2 \
      --raft checkpoints/raft/raft-things.pth \
      [--threshold 0.2,0.35,0.5]
"""

import os
import sys
import glob
import json
import argparse

import numpy as np
import cv2
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_VGGT_REPO = os.path.join(_SCRIPT_DIR, "vggt")
if _VGGT_REPO not in sys.path:
    sys.path.insert(0, _VGGT_REPO)


# ── Shared colormap / panel utilities ────────────────────────────────────────

def _depth_to_color(depth: np.ndarray, vmin: float = None, vmax: float = None) -> np.ndarray:
    if vmin is None:
        vmin = float(np.percentile(depth[depth > 0], 2)) if (depth > 0).any() else 0.0
    if vmax is None:
        vmax = float(np.percentile(depth[depth > 0], 98)) if (depth > 0).any() else 1.0
    norm = np.clip((depth - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def _conf_to_color(conf: np.ndarray) -> np.ndarray:
    mn, mx = float(conf.min()), float(conf.max())
    norm = (conf - mn) / (mx - mn + 1e-8)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)


def _flow_to_color(flow: np.ndarray, max_mag: float = None) -> np.ndarray:
    fx, fy = flow[0], flow[1]
    mag = np.sqrt(fx**2 + fy**2)
    ang = np.arctan2(fy, fx)
    if max_mag is None or max_mag <= 0:
        max_mag = float(np.percentile(mag, 99)) + 1e-6
    h = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    s = np.full_like(h, 255)
    v = (np.clip(mag / max_mag, 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(np.stack([h, s, v], axis=-1), cv2.COLOR_HSV2BGR)


def _residual_to_color(residual: np.ndarray) -> np.ndarray:
    p99 = float(np.percentile(residual, 99)) + 1e-6
    norm = np.clip(residual / p99, 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def _mask_overlay(rgb_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = rgb_bgr.copy()
    if mask.any():
        red = np.zeros_like(out)
        red[:, :, 2] = 255
        out[mask] = (out[mask] * (1 - alpha) + red[mask] * alpha).astype(np.uint8)
    return out


def _add_label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,    1, cv2.LINE_AA)
    return out


def _na_panel(H: int, W: int, text: str) -> np.ndarray:
    out = np.full((H, W, 3), 40, dtype=np.uint8)
    (tw, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.putText(out, text, (max(0, (W - tw) // 2), max(0, (H + th_) // 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    return out


def _residual_stats(residual: np.ndarray, mask: np.ndarray = None) -> dict:
    r = residual[mask] if mask is not None and mask.any() else residual
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {}
    return {"mean": float(r.mean()), "median": float(np.median(r)),
            "p95": float(np.percentile(r, 95)), "max": float(r.max())}


def _preprocess_for_raft(image_paths, mode: str, H: int, W: int) -> np.ndarray:
    """Return [S, H, W, 3] float32 in [0,1] matching VGGT pixel space."""
    import torch as _th
    from PIL import Image as _PIL
    import torchvision.transforms.functional as _TF

    if mode == "letterbox":
        from vggt.utils.load_fn import load_and_preprocess_images_square
        t, _ = load_and_preprocess_images_square(image_paths, target_size=H)
        return t.permute(0, 2, 3, 1).numpy().astype(np.float32)

    imgs = []
    for p in image_paths:
        img = _PIL.open(p).convert("RGB")
        w, h = img.size
        if mode == "center_crop":
            sq = min(w, h)
            img = img.crop(((w-sq)//2, (h-sq)//2, (w+sq)//2, (h+sq)//2))
            img = img.resize((H, H), _PIL.Resampling.BICUBIC)
        elif mode == "long_edge":
            tl = max(H, W)
            nw = tl if w >= h else round(w * tl / h / 14) * 14
            nh = tl if h > w  else round(h * tl / w / 14) * 14
            nw, nh = max(nw, 14), max(nh, 14)
            img = img.resize((nw, nh), _PIL.Resampling.BICUBIC)
        t = _TF.to_tensor(img)
        if t.shape[1] != H or t.shape[2] != W:
            t = _TF.resize(t, [H, W], interpolation=_TF.InterpolationMode.BILINEAR,
                            antialias=True)
        imgs.append(t)
    return _th.stack(imgs).permute(0, 2, 3, 1).numpy().astype(np.float32)


# ── inspect: data helpers ─────────────────────────────────────────────────────

def _load_run_dir(run_dir: str):
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    depth_paths = sorted(glob.glob(os.path.join(run_dir, "depth", "*.npy")))
    mask_paths  = sorted(glob.glob(os.path.join(run_dir, "dynamic_mask", "*.npy")))
    if not depth_paths:
        raise FileNotFoundError(f"No depth/*.npy in {run_dir}")
    depth      = np.stack([np.load(p) for p in depth_paths])
    extrinsics = np.load(os.path.join(run_dir, "extrinsics.npy"))
    intrinsics = np.load(os.path.join(run_dir, "intrinsics.npy"))
    dyn_mask   = (np.stack([np.load(p).astype(bool) for p in mask_paths])
                  if mask_paths else np.zeros(depth.shape, dtype=bool))
    return depth, extrinsics, intrinsics, dyn_mask


def _run_vggt(image_paths, ckpt_path, device):
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt_dyn.initializer import VGGTInitializer

    model = VGGT()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    imgs, _ = load_and_preprocess_images_square(image_paths, target_size=518)
    imgs = imgs.unsqueeze(0).to(device)
    dtype_str = device.type if hasattr(device, "type") else str(device).split(":")[0]
    with torch.no_grad(), torch.autocast(dtype_str, dtype=torch.bfloat16):
        out = model(imgs)

    H, W = out["depth"].shape[2], out["depth"].shape[3]
    init = VGGTInitializer(out, (H, W))
    depth      = init.depth.cpu().numpy()
    depth_conf = init.depth_conf.cpu().numpy()
    R = init.R.cpu().numpy()
    T = init.T.cpu().numpy()
    K = init.K.cpu().numpy()
    extrinsics = np.concatenate([R, T[:, :, None]], axis=-1)
    del model; torch.cuda.empty_cache()
    return depth, depth_conf, extrinsics, K, H, W


def _ego_flow_np(depth, extrinsics, intrinsics, device_str):
    from vggt_dyn.dynamic_mask import compute_ego_flow
    dev = torch.device(device_str)
    return compute_ego_flow(
        torch.from_numpy(depth).float().to(dev),
        torch.from_numpy(extrinsics[:, :3, :3]).float().to(dev),
        torch.from_numpy(extrinsics[:, :3, 3]).float().to(dev),
        torch.from_numpy(intrinsics).float().to(dev),
    ).cpu().numpy()


def _abs_rel(pred, gt, mask=None):
    valid = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if mask is not None:
        valid = valid & mask
    return float(np.mean(np.abs(pred[valid] - gt[valid]) / gt[valid])) if valid.any() else float("nan")


def _make_inspect_panel(rgb_np, depth_np, depth_conf_np, raft_flow_np, ego_flow_np,
                         residual_np, dyn_mask_np, frame_idx, stats, max_flow_mag=None):
    H, W = depth_np.shape
    rgb_u8  = (rgb_np * 255).clip(0, 255).astype(np.uint8) if rgb_np.dtype != np.uint8 else rgb_np
    rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    if rgb_bgr.shape[:2] != (H, W):
        rgb_bgr = cv2.resize(rgb_bgr, (W, H))

    p_rgb   = _add_label(rgb_bgr,                               f"RGB [{frame_idx:04d}]")
    p_depth = _add_label(_depth_to_color(depth_np),             "VGGT Depth")
    p_conf  = _add_label(_conf_to_color(depth_conf_np),         "VGGT Conf")
    p_raft  = _add_label(_flow_to_color(raft_flow_np, max_flow_mag), "RAFT Flow")
    p_ego   = _add_label(_flow_to_color(ego_flow_np,  max_flow_mag), "Ego-Flow")
    p_resid = _add_label(_residual_to_color(residual_np),       "Residual")

    dyn_frac = float(dyn_mask_np.mean()) * 100
    p_mask  = _add_label(_mask_overlay(rgb_bgr, dyn_mask_np),
                          f"Dyn mask {dyn_frac:.1f}%", color=(0, 80, 255))

    for i, (k, v) in enumerate(stats.items()):
        text = f"{k}:{v:.2f}" if isinstance(v, float) else f"{k}:{v}"
        cv2.putText(p_resid, text, (6, 44 + 18*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(p_resid, text, (6, 44 + 18*i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255,255,0), 1, cv2.LINE_AA)

    row0 = np.concatenate([p_rgb, p_depth, p_conf],  axis=1)
    row1 = np.concatenate([p_raft, p_ego,  p_resid], axis=1)
    row2 = np.concatenate([p_mask, np.zeros_like(p_ego), np.zeros_like(p_resid)], axis=1)
    return np.concatenate([row0, row1, row2], axis=0)


# ── gif subcommand ────────────────────────────────────────────────────────────

def gif_main(args):
    import imageio.v2 as imageio

    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    out_dir = args.output_dir

    metrics = {}
    if os.path.isfile(os.path.join(out_dir, "metrics.json")):
        with open(os.path.join(out_dir, "metrics.json")) as f:
            metrics = json.load(f)

    if args.images:
        image_paths = sorted(glob.glob(args.images))
        if not image_paths:
            raise FileNotFoundError(f"No images: {args.images}")
    else:
        rel = metrics.get("image_paths")
        if not rel:
            raise ValueError("No --images and no image_paths in metrics.json")
        image_paths = [os.path.normpath(os.path.join(out_dir, p.replace("\\", "/"))) for p in rel]

    depth_paths = sorted(glob.glob(os.path.join(out_dir, "depth", "*.npy")))
    mask_paths  = sorted(glob.glob(os.path.join(out_dir, "dynamic_mask", "*.npy")))
    conf_paths  = sorted(glob.glob(os.path.join(out_dir, "depth_conf", "*.npy")))
    if not depth_paths:
        raise FileNotFoundError(f"No depth/*.npy in {out_dir}")

    depth      = np.stack([np.load(p) for p in depth_paths])
    extrinsics = np.load(os.path.join(out_dir, "extrinsics.npy"))
    intrinsics = np.load(os.path.join(out_dir, "intrinsics.npy"))
    dyn_mask_pre = (np.stack([np.load(p).astype(bool) for p in mask_paths])
                    if mask_paths else np.zeros(depth.shape, dtype=bool))
    has_conf   = len(conf_paths) == depth.shape[0]
    depth_conf = np.stack([np.load(p) for p in conf_paths]) if has_conf else None

    S = min(depth.shape[0], len(image_paths))
    if args.max_frames:
        S = min(S, args.max_frames)
    image_paths = image_paths[:S]
    depth, extrinsics, intrinsics, dyn_mask_pre = (
        depth[:S], extrinsics[:S], intrinsics[:S], dyn_mask_pre[:S])
    if depth_conf is not None:
        depth_conf = depth_conf[:S]
    H, W = depth.shape[1], depth.shape[2]
    print(f"[gif] {S} frames, depth ({H}×{W})")

    images_bgr = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {p}")
        images_bgr.append(cv2.resize(img, (W, H)))

    vmin = float(np.percentile(depth.ravel(), 2))
    vmax = float(np.percentile(depth.ravel(), 98))

    flow_fwd = ego_flow = residual = valid_fwd = dyn_mask_computed = max_mag = None
    if args.raft:
        mode = metrics.get("preprocess", "letterbox")
        print(f"[gif] preprocessing for RAFT ({mode}) ...")
        images_np = _preprocess_for_raft(image_paths, mode, H, W)
        from vggt_dyn.utils.flow_utils import compute_adjacent_flow
        flow_fwd_t, _, valid_fwd_t, _ = compute_adjacent_flow(
            list(images_np), model_path=args.raft,
            chunk_size=args.chunk_size, device=device_str)
        flow_fwd  = flow_fwd_t.cpu().numpy()
        valid_fwd = valid_fwd_t.cpu().numpy()
        from vggt_dyn.dynamic_mask import compute_ego_flow, build_dynamic_mask, pair_mask_to_frame_mask
        ego_flow = compute_ego_flow(
            torch.from_numpy(depth).float().to(device),
            torch.from_numpy(extrinsics[:, :3, :3]).float().to(device),
            torch.from_numpy(extrinsics[:, :3, 3]).float().to(device),
            torch.from_numpy(intrinsics).float().to(device),
        ).cpu().numpy()
        diff     = ego_flow - flow_fwd
        residual = np.sqrt((diff**2).sum(axis=1))
        residual[~valid_fwd[:, 0].astype(bool)] = 0.0
        pair_mask       = build_dynamic_mask(
            torch.from_numpy(residual).unsqueeze(1), threshold=args.threshold, normalize=True)
        dyn_mask_computed = pair_mask_to_frame_mask(pair_mask, S)[:, 0].numpy()
        max_mag = float(np.percentile(np.sqrt((flow_fwd**2).sum(axis=1)), 99))

    print(f"[gif] generating {S} panels ...")
    all_stats, gif_frames = [], []
    for i in range(S):
        rgb = images_bgr[i]
        dm  = dyn_mask_computed[i] if dyn_mask_computed is not None else dyn_mask_pre[i]

        p_rgb   = _add_label(rgb,                              f"RGB [{i:04d}]")
        p_depth = _add_label(_depth_to_color(depth[i], vmin, vmax), "Depth")
        p_mask  = _add_label(_mask_overlay(rgb, dm),
                              f"Dyn Mask {dm.mean()*100:.1f}%", color=(0, 80, 255))
        p_conf  = (_add_label(_conf_to_color(depth_conf[i]),
                               f"Conf [{depth_conf[i].min():.3f},{depth_conf[i].max():.3f}]")
                   if depth_conf is not None else _na_panel(H, W, "Conf: N/A"))

        fs = {"frame": i, "dyn_pct": round(float(dm.mean()) * 100, 2)}
        if flow_fwd is not None:
            pi = min(i, S-2)
            rf, ef, rs = flow_fwd[pi], ego_flow[pi], residual[pi]
            p_raft = _add_label(_flow_to_color(rf, max_mag), "RAFT Flow")
            p_ego  = _add_label(_flow_to_color(ef, max_mag), "Ego-Flow")
            st     = _residual_stats(rs, valid_fwd[pi, 0].astype(bool))
            fs.update({"raft_mean": float(np.sqrt((rf**2).sum(0)).mean()),
                        "ego_mean":  float(np.sqrt((ef**2).sum(0)).mean()),
                        "resid_median": st.get("median", float("nan")),
                        "resid_p95":    st.get("p95",    float("nan"))})
        else:
            p_raft = _na_panel(H, W, "RAFT Flow: N/A")
            p_ego  = _na_panel(H, W, "Ego-Flow: N/A")
        all_stats.append(fs)
        panel = np.concatenate([
            np.concatenate([p_rgb, p_depth, p_mask], axis=1),
            np.concatenate([p_conf, p_raft, p_ego],  axis=1),
        ], axis=0)
        gif_frames.append(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))

    imageio.mimsave(os.path.join(out_dir, "viz.gif"), gif_frames,
                    duration=1.0/args.fps, loop=0)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"[gif] saved → {out_dir}/viz.gif")


# ── inspect subcommand ────────────────────────────────────────────────────────

def inspect_main(args):
    device_str = args.device if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)
    print(f"[inspect] device = {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "panels"), exist_ok=True)

    thresholds = [float(t) for t in args.threshold.split(",")]

    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        raise FileNotFoundError(f"No images: {args.images}")
    S = len(image_paths) if not args.max_frames else min(args.max_frames, len(image_paths))
    image_paths = image_paths[:S]
    print(f"[inspect] {S} frames, thresholds={thresholds}")

    depth_conf = None
    if args.from_run:
        depth, extrinsics, intrinsics, dyn_mask_pre = _load_run_dir(args.from_run)
        depth, extrinsics, intrinsics = depth[:S], extrinsics[:S], intrinsics[:S]
        dyn_mask_pre = dyn_mask_pre[:S]
        H, W = depth.shape[1], depth.shape[2]
    elif args.ckpt:
        depth, depth_conf, extrinsics, intrinsics, H, W = _run_vggt(image_paths, args.ckpt, device)
        depth, extrinsics, intrinsics = depth[:S], extrinsics[:S], intrinsics[:S]
        depth_conf = depth_conf[:S] if depth_conf is not None else None
        dyn_mask_pre = None
    else:
        raise ValueError("Provide --from_run or --ckpt")

    if depth_conf is None:
        depth_conf = np.ones_like(depth)

    mode = "letterbox"
    if args.from_run:
        mp = os.path.join(args.from_run, "metrics.json")
        if os.path.isfile(mp):
            with open(mp) as f:
                mode = json.load(f).get("preprocess", "letterbox")
    if args.preprocess:
        mode = args.preprocess
    print(f"[inspect] preprocess={mode}")

    print("[inspect] preprocessing images for RAFT ...")
    images_np = _preprocess_for_raft(image_paths, mode, H, W)

    print("[inspect] computing RAFT flow ...")
    from vggt_dyn.utils.flow_utils import compute_adjacent_flow
    flow_fwd_t, _, valid_fwd_t, _ = compute_adjacent_flow(
        list(images_np), model_path=args.raft,
        chunk_size=args.chunk_size, device=device_str)
    flow_fwd  = flow_fwd_t.cpu().numpy()
    valid_fwd = valid_fwd_t.cpu().numpy()

    print("[inspect] computing ego-flow ...")
    ego_flow = _ego_flow_np(depth, extrinsics, intrinsics, device_str)

    diff     = ego_flow - flow_fwd
    residual = np.sqrt((diff**2).sum(axis=1))
    valid_np = valid_fwd[:, 0]
    residual[~valid_np.astype(bool)] = 0.0

    from vggt_dyn.dynamic_mask import build_dynamic_mask, pair_mask_to_frame_mask
    resid_t   = torch.from_numpy(residual).unsqueeze(1)
    dyn_masks = {}
    for thr in thresholds:
        pm = build_dynamic_mask(resid_t, threshold=thr, normalize=True)
        dyn_masks[thr] = pair_mask_to_frame_mask(pm, S)[:, 0].numpy()

    primary_thr       = thresholds[0]
    dyn_mask_computed = dyn_masks[primary_thr]
    if dyn_mask_pre is not None:
        match = float(np.mean(dyn_mask_pre == dyn_mask_computed)) * 100
        print(f"[inspect] pre-saved vs recomputed mask: {match:.1f}% agreement")

    gt_depth_global = None
    if args.gt_depth:
        ext = os.path.splitext(args.gt_depth)[1].lower()
        gt_raw = (np.array(__import__("imageio").imread(args.gt_depth), dtype=np.float32)
                  if ext in (".tif", ".tiff") else np.load(args.gt_depth).astype(np.float32))
        if gt_raw.ndim == 2:
            gt_depth_global = cv2.resize(gt_raw, (W, H), interpolation=cv2.INTER_NEAREST)
            print(f"[inspect] GT Abs Rel (frame 0): {_abs_rel(depth[0], gt_depth_global):.4f}")

    max_mag     = float(np.percentile(np.sqrt((flow_fwd**2).sum(axis=1)), 99))
    all_stats   = []
    panel_paths = []
    print("[inspect] generating panels ...")

    for t in range(S):
        pi  = min(t, S-2)
        rf  = flow_fwd[pi]
        ef  = ego_flow[pi]
        rs  = residual[pi]
        dm  = dyn_mask_computed[t]
        st  = _residual_stats(rs, valid_np[pi])
        fs  = {
            "frame":        t,
            "dyn_%":        round(float(dm.mean()) * 100, 2),
            "raft_mean":    round(float(np.sqrt((rf**2).sum(0)).mean()), 3),
            "ego_mean":     round(float(np.sqrt((ef**2).sum(0)).mean()), 3),
            "resid_mean":   round(st.get("mean",   float("nan")), 3),
            "resid_median": round(st.get("median", float("nan")), 3),
            "resid_p95":    round(st.get("p95",    float("nan")), 3),
        }
        if gt_depth_global is not None and t == 0:
            fs["abs_rel_gt"] = round(_abs_rel(depth[t], gt_depth_global), 4)
        if len(thresholds) > 1:
            fs["threshold_sweep"] = {
                f"dyn%_thr={thr:.2f}": round(float(dyn_masks[thr][t].mean()) * 100, 1)
                for thr in thresholds}
        all_stats.append(fs)

        panel = _make_inspect_panel(
            images_np[t], depth[t], depth_conf[t],
            rf, ef, rs, dm, t,
            {"dyn%": fs["dyn_%"], "r_med": st.get("median", float("nan")),
             "r_p95": st.get("p95", float("nan"))},
            max_mag)
        out_path = os.path.join(args.save_dir, "panels", f"{t:04d}.png")
        cv2.imwrite(out_path, panel)
        panel_paths.append(out_path)
        if t % 10 == 0 or t == S-1:
            print(f"  frame {t:3d}/{S-1}  dyn={fs['dyn_%']:5.1f}%  "
                  f"resid_med={st.get('median', float('nan')):.2f}")

    with open(os.path.join(args.save_dir, "stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"[inspect] stats → {args.save_dir}/stats.json")

    dyn_fracs  = [s["dyn_%"] for s in all_stats]
    resid_meds = [s["resid_median"] for s in all_stats if not np.isnan(s["resid_median"])]
    print(f"\n── Summary: {S} frames  dyn%={np.mean(dyn_fracs):.1f}%  "
          f"resid_median={np.mean(resid_meds):.2f}px")
    for thr in thresholds:
        fracs = [float(dyn_masks[thr][t].mean()) * 100 for t in range(S)]
        print(f"  thr={thr:.2f} → dyn%={np.mean(fracs):.1f}%")

    if args.gif:
        try:
            import imageio.v2 as iio
            iio.mimwrite(os.path.join(args.save_dir, "viz.gif"),
                         [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in panel_paths],
                         fps=5, loop=0)
            print(f"[inspect] GIF → {args.save_dir}/viz.gif")
        except Exception as e:
            print(f"[inspect] GIF failed: {e}")

    print(f"[inspect] done → {args.save_dir}/panels/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _add_common_run_opts(p):
    p.add_argument("--device",     default="cuda")
    p.add_argument("--chunk_size", type=int, default=8)
    p.add_argument("--max_frames", type=int, default=None)


def parse_args():
    top = argparse.ArgumentParser(
        description="VGGT-Dyn diagnostic visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = top.add_subparsers(dest="subcommand", required=True)

    # ── gif ───────────────────────────────────────────────────────────────────
    gp = sub.add_parser("gif", help="quick GIF: RGB/Depth/Mask (+ optional RAFT/Ego-flow)")
    gp.add_argument("--output_dir", required=True, help="run.py --output directory")
    gp.add_argument("--images",    default=None,
                    help="glob for input images; if omitted, read from metrics.json")
    gp.add_argument("--raft",      default=None, help="RAFT checkpoint (.pth)")
    gp.add_argument("--threshold", type=float, default=0.35)
    gp.add_argument("--fps",       type=int, default=5)
    _add_common_run_opts(gp)

    # ── inspect ───────────────────────────────────────────────────────────────
    ip = sub.add_parser("inspect", help="detailed 6-panel: ego-flow, residual, threshold sweep")
    ip.add_argument("--images",    required=True, help="glob for input frames")
    ip.add_argument("--raft",      required=True, help="RAFT checkpoint (.pth)")
    ip.add_argument("--save_dir",  default="outputs/inspect")
    ip.add_argument("--from_run",  default=None, help="[Mode A] existing run.py output dir")
    ip.add_argument("--ckpt",      default=None, help="[Mode B] VGGT checkpoint (.pt)")
    ip.add_argument("--gt_depth",  default=None, help="GT depth (.tiff/.npy) for AbsRel eval")
    ip.add_argument("--threshold", default="0.35",
                    help="comma-separated thresholds, e.g. '0.2,0.35,0.5'")
    ip.add_argument("--preprocess", default=None,
                    choices=["letterbox", "center_crop", "long_edge"])
    ip.add_argument("--gif", action="store_true", help="also save viz.gif")
    _add_common_run_opts(ip)

    return top.parse_args()


def main():
    args = parse_args()
    if args.subcommand == "gif":
        gif_main(args)
    else:
        inspect_main(args)


if __name__ == "__main__":
    main()
