#!/usr/bin/env python3
"""
inspect_components.py — 視覺化 VGGT depth/pose、RAFT flow、ego-flow、
flow residual、dynamic mask，用於快速診斷各元件是否正確。

兩種使用模式
─────────────
Mode A（快速）：載入已有的 run.py 輸出，重算 RAFT flow 做比對
    python inspect_components.py \
        --images   "data/SCARED/dataset_8/keyframe_0/data/frames/*.png" \
        --from_run outputs/vggt_dyn_d8k0 \
        --raft     monst3r/third_party/RAFT/models/raft-things.pth \
        --save_dir outputs/inspect_d8k0

Mode B（完整）：從頭跑 VGGT 再做診斷
    python inspect_components.py \
        --images   "data/SCARED/dataset_8/keyframe_0/data/frames/*.png" \
        --ckpt     vggt/checkpoints/VGGT-1B.pt \
        --raft     monst3r/third_party/RAFT/models/raft-things.pth \
        --save_dir outputs/inspect_d8k0

選項
─────
--gt_depth    GT 深度圖路徑（.tiff 或 .npy），支援 keyframe-level 評估
--threshold   dynamic mask 的 flow residual 閾值，可以多個值「,」分隔
              例如 0.2,0.35,0.5（預設 0.35）
--max_frames  最多視覺化幾幀（預設全部）
--device      cuda / cpu

輸出（儲存至 --save_dir）
──────────────────────────
panels/XXXX.png     每幀 6-panel 並排圖
stats.json          每幀統計數字
viz.gif             整段動圖（可選）
"""

import os
import sys
import glob
import json
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_VGGT_REPO = os.path.join(_SCRIPT_DIR, "vggt")
if _VGGT_REPO not in sys.path:
    sys.path.insert(0, _VGGT_REPO)

import torch
import torch.nn.functional as F
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Flow → colormap  (HSV colorwheel，同 Middlebury)
# ─────────────────────────────────────────────────────────────────────────────

def flow_to_color(flow: np.ndarray, max_mag: float = None) -> np.ndarray:
    """Convert [2, H, W] float32 flow → [H, W, 3] BGR uint8 via HSV colorwheel.

    Hue = direction, Value = magnitude (clipped to max_mag).
    """
    fx, fy = flow[0], flow[1]
    mag = np.sqrt(fx**2 + fy**2)
    ang = np.arctan2(fy, fx)  # [-π, π]

    if max_mag is None or max_mag <= 0:
        max_mag = np.percentile(mag, 99) + 1e-6

    # Hue [0, 180] for OpenCV HSV
    h = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    s = np.full_like(h, 255)
    v = (np.clip(mag / max_mag, 0, 1) * 255).astype(np.uint8)

    hsv = np.stack([h, s, v], axis=-1)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def residual_to_color(residual: np.ndarray) -> np.ndarray:
    """[H, W] float → [H, W, 3] BGR with HOT colormap (blue=low, red=high)."""
    p99 = np.percentile(residual, 99) + 1e-6
    norm = np.clip(residual / p99, 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_HOT)


def depth_to_color(depth: np.ndarray) -> np.ndarray:
    """[H, W] float → [H, W, 3] BGR with TURBO colormap."""
    p2  = np.percentile(depth[depth > 0], 2)  if (depth > 0).any() else 0
    p98 = np.percentile(depth[depth > 0], 98) if (depth > 0).any() else 1
    norm = np.clip((depth - p2) / (p98 - p2 + 1e-8), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def conf_to_color(conf: np.ndarray) -> np.ndarray:
    """[H, W] float in any range → gray-scale BGR."""
    mn, mx = conf.min(), conf.max()
    norm = (conf - mn) / (mx - mn + 1e-8)
    u8   = (norm * 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


def mask_overlay_bgr(rgb_bgr: np.ndarray, mask: np.ndarray,
                     alpha: float = 0.55) -> np.ndarray:
    """Overlay bool mask as semi-transparent red on BGR image."""
    out = rgb_bgr.copy()
    if mask.any():
        red = np.zeros_like(out)
        red[:, :, 2] = 255
        out[mask] = (out[mask] * (1 - alpha) + red[mask] * alpha).astype(np.uint8)
    return out


def label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color,   1, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_from_run_dir(run_dir: str):
    """Load VGGT-Dyn run.py outputs.

    Returns:
        depth       : [S, H, W] float32
        extrinsics  : [S, 3, 4] float32
        intrinsics  : [S, 3, 3] float32
        dyn_mask    : [S, H, W] bool  (per-frame)
    """
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        parent = os.path.dirname(run_dir)
        hint = ""
        if os.path.isdir(parent):
            cand = [
                d for d in sorted(os.listdir(parent))
                if os.path.isdir(os.path.join(parent, d))
            ]
            if cand:
                hint = f" Available under {parent}: {', '.join(cand[:8])}"
        raise FileNotFoundError(f"run_dir not found: {run_dir}.{hint}")

    depth_paths = sorted(glob.glob(os.path.join(run_dir, "depth", "*.npy")))
    mask_paths  = sorted(glob.glob(os.path.join(run_dir, "dynamic_mask", "*.npy")))
    ext_path    = os.path.join(run_dir, "extrinsics.npy")
    intr_path   = os.path.join(run_dir, "intrinsics.npy")

    if not depth_paths:
        has_depth_orig = bool(sorted(glob.glob(os.path.join(run_dir, "depth_orig_res", "*.npy"))))
        msg = f"No depth/*.npy found in {run_dir}"
        if has_depth_orig:
            msg += (
                "; found depth_orig_res/*.npy only. "
                "This usually means the source is not a full run.py output for inspect (missing depth/)."
            )
        raise FileNotFoundError(msg)
    if not os.path.isfile(ext_path):
        raise FileNotFoundError(f"extrinsics.npy not found in {run_dir}")
    if not os.path.isfile(intr_path):
        raise FileNotFoundError(f"intrinsics.npy not found in {run_dir}")

    depth = np.stack([np.load(p) for p in depth_paths], axis=0)   # [S, H, W]
    extrinsics = np.load(ext_path)   # [S, 3, 4]
    intrinsics = np.load(intr_path)  # [S, 3, 3]

    if mask_paths:
        dyn_mask = np.stack([np.load(p).astype(bool) for p in mask_paths], axis=0)
    else:
        dyn_mask = np.zeros(depth.shape, dtype=bool)

    return depth, extrinsics, intrinsics, dyn_mask


def run_vggt_fresh(image_paths, ckpt_path, device):
    """Run VGGT feed-forward and return parsed components."""
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt_dyn.initializer import VGGTInitializer

    model = VGGT()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()

    imgs_tensor, _ = load_and_preprocess_images_square(image_paths, target_size=518)
    imgs_tensor = imgs_tensor.unsqueeze(0).to(device)

    dtype = device.type if hasattr(device, "type") else str(device).split(":")[0]
    with torch.no_grad(), torch.autocast(dtype, dtype=torch.bfloat16):
        vggt_out = model(imgs_tensor)

    H = vggt_out["depth"].shape[2]
    W = vggt_out["depth"].shape[3]
    init = VGGTInitializer(vggt_out, (H, W))

    depth      = init.depth.cpu().numpy()          # [S, H, W]
    depth_conf = init.depth_conf.cpu().numpy()     # [S, H, W]

    R = init.R.cpu().numpy()   # [S, 3, 3]
    T = init.T.cpu().numpy()   # [S, 3]
    K = init.K.cpu().numpy()   # [S, 3, 3]

    S = R.shape[0]
    extrinsics = np.concatenate([R, T[:, :, None]], axis=-1)  # [S, 3, 4]

    del model
    torch.cuda.empty_cache()

    return depth, depth_conf, extrinsics, K, H, W


# ─────────────────────────────────────────────────────────────────────────────
# Ego-flow computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ego_flow_np(depth: np.ndarray,
                        extrinsics: np.ndarray,
                        intrinsics: np.ndarray,
                        device_str: str = "cpu") -> np.ndarray:
    """Wrapper: call dynamic_mask.compute_ego_flow with numpy arrays.

    Args:
        depth      : [S, H, W]
        extrinsics : [S, 3, 4]  (R|T)
        intrinsics : [S, 3, 3]
        device_str : 'cuda' or 'cpu'

    Returns:
        ego_flow : [S-1, 2, H, W] float32 numpy
    """
    from vggt_dyn.dynamic_mask import compute_ego_flow

    dev = torch.device(device_str)
    d_t   = torch.from_numpy(depth).float().to(dev)          # [S, H, W]
    R_t   = torch.from_numpy(extrinsics[:, :3, :3]).float().to(dev)  # [S, 3, 3]
    T_t   = torch.from_numpy(extrinsics[:, :3, 3]).float().to(dev)   # [S, 3]
    K_t   = torch.from_numpy(intrinsics).float().to(dev)     # [S, 3, 3]

    ego = compute_ego_flow(d_t, R_t, T_t, K_t)  # [S-1, 2, H, W]
    return ego.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def abs_rel(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None) -> float:
    """Compute Abs Rel = mean(|pred - gt| / gt) on valid pixels."""
    valid = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if mask is not None:
        valid = valid & mask
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - gt[valid]) / gt[valid]))


def compute_residual_stats(residual: np.ndarray, mask: np.ndarray = None) -> dict:
    """[H, W] residual → stats dict."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Build one 6-panel frame figure
# ─────────────────────────────────────────────────────────────────────────────

def make_panel(rgb_np: np.ndarray,
               depth_np: np.ndarray,
               depth_conf_np: np.ndarray,
               raft_flow_np: np.ndarray,   # [2, H, W]
               ego_flow_np: np.ndarray,    # [2, H, W]
               residual_np: np.ndarray,    # [H, W]
               dyn_mask_np: np.ndarray,    # [H, W] bool
               frame_idx: int,
               stats: dict,
               max_flow_mag: float = None) -> np.ndarray:
    """Build a 2×3 panel figure.

    Row 0: [RGB | VGGT Depth | VGGT Conf]
    Row 1: [RAFT flow | Ego-flow | Residual | Mask overlay]
      (mask overlay 擠進第 2 行第 3 格)
    """
    H, W = depth_np.shape

    # ── Convert letterboxed RGB to uint8 BGR ─────────────────────────────────
    # images_np is already (H, W, 3) float32 [0,1] in RGB (letterboxed)
    if rgb_np.dtype != np.uint8:
        rgb_u8 = (rgb_np * 255).clip(0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb_np
    rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    if rgb_bgr.shape[:2] != (H, W):
        rgb_bgr = cv2.resize(rgb_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    # ── Build 6 panels ────────────────────────────────────────────────────────
    p_rgb    = label(rgb_bgr,                         f"RGB  [{frame_idx:04d}]")
    p_depth  = label(depth_to_color(depth_np),        "VGGT Depth")
    p_conf   = label(conf_to_color(depth_conf_np),    "VGGT Conf")
    p_raft   = label(flow_to_color(raft_flow_np, max_flow_mag), "RAFT Flow")
    p_ego    = label(flow_to_color(ego_flow_np,  max_flow_mag), "Ego-Flow")
    p_resid  = label(residual_to_color(residual_np),  "Residual")

    # mask panel = residual heatmap with mask contour
    dyn_frac = float(dyn_mask_np.mean()) * 100
    p_mask   = label(mask_overlay_bgr(rgb_bgr, dyn_mask_np),
                     f"Dyn mask {dyn_frac:.1f}%", color=(0, 80, 255))

    # ── Stat overlay on residual panel ────────────────────────────────────────
    for i, (k, v) in enumerate(stats.items()):
        text = f"{k}:{v:.2f}" if isinstance(v, float) else f"{k}:{v}"
        cv2.putText(p_resid, text, (6, 44 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0),   2, cv2.LINE_AA)
        cv2.putText(p_resid, text, (6, 44 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    row0 = np.concatenate([p_rgb, p_depth, p_conf],  axis=1)
    row1 = np.concatenate([p_raft, p_ego, p_resid],  axis=1)
    row2 = np.concatenate([p_mask,
                           np.zeros_like(p_ego),      # blank
                           np.zeros_like(p_resid)],   # blank
                          axis=1)

    return np.concatenate([row0, row1, row2], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("--images", required=True,
                   help="glob pattern for input frames, e.g. 'frames/*.png'")
    p.add_argument("--raft",   required=True,
                   help="RAFT checkpoint (.pth)")
    p.add_argument("--save_dir", default="outputs/inspect",
                   help="output directory for panels + stats")

    # ── Mode A: load from existing run.py output ──────────────────────────────
    p.add_argument("--from_run", default=None,
                   help="[Mode A] path to existing run.py --output directory")

    # ── Mode B: run VGGT fresh ────────────────────────────────────────────────
    p.add_argument("--ckpt", default=None,
                   help="[Mode B] VGGT checkpoint (.pt)")

    # ── Optional GT depth for Abs Rel evaluation ──────────────────────────────
    p.add_argument("--gt_depth", default=None,
                   help="optional GT depth file (.tiff / .npy) for AbsRel eval")

    # ── Thresholds ───────────────────────────────────────────────────────────
    p.add_argument("--threshold", default="0.35",
                   help="comma-separated flow residual thresholds, e.g. '0.2,0.35,0.5'")

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument("--max_frames", type=int, default=None,
                   help="max frames to visualize (default: all)")
    p.add_argument("--preprocess", default=None,
                   choices=["letterbox", "center_crop", "long_edge"],
                   help="override preprocess mode (auto-detected from metrics.json if not set)")
    p.add_argument("--device", default="cuda",
                   help="cuda / cpu")
    p.add_argument("--gif", action="store_true",
                   help="also save viz.gif")
    p.add_argument("--chunk_size", type=int, default=8,
                   help="RAFT batch size per GPU call")

    return p.parse_args()


def main():
    args = parse_args()

    device_str = args.device if torch.cuda.is_available() else "cpu"
    device     = torch.device(device_str)
    print(f"[inspect] device = {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "panels"), exist_ok=True)

    thresholds = [float(t) for t in args.threshold.split(",")]
    print(f"[inspect] dynamic mask thresholds: {thresholds}")

    # ── 1. Load images ────────────────────────────────────────────────────────
    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        raise FileNotFoundError(f"No images found: {args.images}")
    S_total = len(image_paths)
    S = S_total if args.max_frames is None else min(args.max_frames, S_total)
    image_paths = image_paths[:S]
    print(f"[inspect] {S} / {S_total} frames")

    # ── 2. VGGT outputs ───────────────────────────────────────────────────────
    depth_conf = None  # may not be available in Mode A

    if args.from_run:
        print(f"[inspect] Mode A — loading from {args.from_run}")
        depth, extrinsics, intrinsics, dyn_mask_pre = load_from_run_dir(args.from_run)
        depth, extrinsics, intrinsics = (
            depth[:S], extrinsics[:S], intrinsics[:S]
        )
        dyn_mask_pre = dyn_mask_pre[:S]
        H, W = depth.shape[1], depth.shape[2]
    elif args.ckpt:
        print(f"[inspect] Mode B — running VGGT from {args.ckpt}")
        depth, depth_conf, extrinsics, intrinsics, H, W = run_vggt_fresh(
            image_paths, args.ckpt, device
        )
        depth, extrinsics, intrinsics = depth[:S], extrinsics[:S], intrinsics[:S]
        if depth_conf is not None:
            depth_conf = depth_conf[:S]
        dyn_mask_pre = None  # will be computed below
    else:
        raise ValueError("Provide --from_run (Mode A) or --ckpt (Mode B).")

    if depth_conf is None:
        # Mode A: no confidence stored; fill with ones
        depth_conf = np.ones_like(depth)

    print(f"[inspect] depth shape: {depth.shape},  H={H}, W={W}")

    # ── Detect preprocess mode from saved metrics (center_crop vs letterbox) ──
    _preprocess_mode = "letterbox"
    if args.from_run:
        _metrics_path = os.path.join(args.from_run, "metrics.json")
        if os.path.isfile(_metrics_path):
            import json as _json
            with open(_metrics_path) as _f:
                _m = _json.load(_f)
            _preprocess_mode = _m.get("preprocess", "letterbox")
    if hasattr(args, "preprocess") and args.preprocess:
        _preprocess_mode = args.preprocess
    print(f"[inspect] preprocess mode: {_preprocess_mode}")

    # ── 3. Preprocess images for RAFT (same space as VGGT) ───────────────────
    # CRITICAL: RAFT must use the SAME pixel space as VGGT depth/pose.
    print(f"[inspect] preprocessing images for RAFT ({_preprocess_mode}) ...")
    if _preprocess_mode == "center_crop":
        from PIL import Image as _PIL_Image
        import torchvision.transforms.functional as _TF
        _imgs_cc = []
        for _p in image_paths:
            _img = _PIL_Image.open(_p).convert("RGB")
            _w, _h = _img.size
            _sq = min(_w, _h)
            _left, _top = (_w - _sq) // 2, (_h - _sq) // 2
            _img = _img.crop((_left, _top, _left + _sq, _top + _sq))
            _img = _img.resize((H, H), _PIL_Image.Resampling.BICUBIC)
            _imgs_cc.append(_TF.to_tensor(_img))
        imgs_tensor_pp = torch.stack(_imgs_cc)
    elif _preprocess_mode == "long_edge":
        from PIL import Image as _PIL_Image
        import torchvision.transforms.functional as _TF
        _target_long = max(H, W)
        _imgs_le = []
        for _p in image_paths:
            _img = _PIL_Image.open(_p).convert("RGB")
            _w, _h = _img.size
            if _w >= _h:
                _new_w = _target_long
                _new_h = round(_h * (_new_w / _w) / 14) * 14
            else:
                _new_h = _target_long
                _new_w = round(_w * (_new_h / _h) / 14) * 14
            _new_w = max(_new_w, 14)
            _new_h = max(_new_h, 14)
            _img = _img.resize((_new_w, _new_h), _PIL_Image.Resampling.BICUBIC)
            _img_t = _TF.to_tensor(_img)
            # Ensure exact shape match with loaded depth/extrinsics pixel space.
            if _img_t.shape[1] != H or _img_t.shape[2] != W:
                _img_t = _TF.resize(
                    _img_t, [H, W],
                    interpolation=_TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            _imgs_le.append(_img_t)
        imgs_tensor_pp = torch.stack(_imgs_le)
    else:
        from vggt.utils.load_fn import load_and_preprocess_images_square
        imgs_tensor_pp, _ = load_and_preprocess_images_square(image_paths, target_size=H)
    images_np = imgs_tensor_pp.permute(0, 2, 3, 1).numpy()  # [S, H, W, 3] float32 [0,1]

    # ── 4. RAFT flow ──────────────────────────────────────────────────────────
    print("[inspect] computing RAFT flow ...")
    from vggt_dyn.utils.flow_utils import compute_adjacent_flow

    imgs_norm = images_np.astype(np.float32)  # already [0,1] from ToTensor
    flow_fwd_t, flow_bwd_t, valid_fwd_t, _ = compute_adjacent_flow(
        list(imgs_norm),
        model_path=args.raft,
        chunk_size=args.chunk_size,
        device=device_str,
    )
    flow_fwd  = flow_fwd_t.cpu().numpy()   # [S-1, 2, H, W]
    valid_fwd = valid_fwd_t.cpu().numpy()  # [S-1, 1, H, W]

    # ── 5. Ego-flow ───────────────────────────────────────────────────────────
    print("[inspect] computing ego-flow from VGGT depth + pose ...")
    ego_flow = compute_ego_flow_np(depth, extrinsics, intrinsics, device_str)
    # [S-1, 2, H, W]

    # ── 6. Flow residual ──────────────────────────────────────────────────────
    # |ego_flow - raft_flow| magnitude, zeroed at invalid pixels
    diff     = ego_flow - flow_fwd             # [S-1, 2, H, W]
    residual = np.sqrt((diff**2).sum(axis=1))  # [S-1, H, W]
    valid_np = valid_fwd[:, 0, :, :]           # [S-1, H, W]
    residual[~valid_np.astype(bool)] = 0.0

    # ── 7. Dynamic masks at each threshold ───────────────────────────────────
    from vggt_dyn.dynamic_mask import build_dynamic_mask, pair_mask_to_frame_mask
    import torch as th

    resid_t = th.from_numpy(residual).unsqueeze(1)  # [S-1, 1, H, W]

    dyn_masks = {}
    for thr in thresholds:
        pair_mask = build_dynamic_mask(resid_t, threshold=thr, normalize=True)
        frame_mask = pair_mask_to_frame_mask(pair_mask, S)  # [S, 1, H, W]
        dyn_masks[thr] = frame_mask[:, 0].numpy()           # [S, H, W] bool

    # Use the first threshold as the primary display mask
    primary_thr = thresholds[0]
    dyn_mask_computed = dyn_masks[primary_thr]  # [S, H, W]

    # For Mode A: compare pre-saved mask vs freshly computed mask
    if dyn_mask_pre is not None:
        match = np.mean(dyn_mask_pre == dyn_mask_computed) * 100
        print(f"[inspect] pre-saved mask vs recomputed (thr={primary_thr:.2f}): "
              f"pixel agreement = {match:.1f}%")

    # ── 8. Optional GT depth evaluation ──────────────────────────────────────
    gt_depth_global = None
    if args.gt_depth:
        print(f"[inspect] loading GT depth from {args.gt_depth} ...")
        ext = os.path.splitext(args.gt_depth)[1].lower()
        if ext in (".tif", ".tiff"):
            import imageio
            gt_raw = np.array(imageio.imread(args.gt_depth), dtype=np.float32)
        else:
            gt_raw = np.load(args.gt_depth).astype(np.float32)

        # GT may be for keyframe only (single frame)
        if gt_raw.ndim == 2:
            gt_depth_global = cv2.resize(gt_raw, (W, H),
                                         interpolation=cv2.INTER_NEAREST)
            ar = abs_rel(depth[0], gt_depth_global)
            print(f"[inspect] GT Abs Rel (frame 0, static mask): {ar:.4f}")

    # ── 9. Global stats ───────────────────────────────────────────────────────
    all_stats  = []
    # Compute shared max flow magnitude for consistent colormap across frames
    max_mag = float(np.percentile(np.sqrt((flow_fwd**2).sum(axis=1)), 99))

    # ── 10. Per-frame visualization ───────────────────────────────────────────
    print("[inspect] generating panels ...")
    panel_paths = []

    for t in range(S):
        # pair index: for frame t use pair (t-1,t) or (t,t+1)
        # frame 0 has no pair before it, use pair 0 (t→t+1)
        # frame S-1 has no pair after, use pair S-2
        pair_idx = min(t, S - 2)

        rf = flow_fwd[pair_idx]       # [2, H, W]
        ef = ego_flow[pair_idx]       # [2, H, W]
        rs = residual[pair_idx]       # [H, W]
        dm = dyn_mask_computed[t]     # [H, W] bool

        # stats for this pair
        frac_dyn = float(dm.mean()) * 100
        raft_mag = float(np.sqrt((rf**2).sum(axis=0)).mean())
        ego_mag  = float(np.sqrt((ef**2).sum(axis=0)).mean())
        r_stats  = compute_residual_stats(rs, valid_np[pair_idx])

        per_frame = {
            "frame":     t,
            "dyn_%":     round(frac_dyn, 2),
            "raft_mean": round(raft_mag, 3),
            "ego_mean":  round(ego_mag,  3),
            "resid_mean":   round(r_stats.get("mean",  float("nan")), 3),
            "resid_median": round(r_stats.get("median",float("nan")), 3),
            "resid_p95":    round(r_stats.get("p95",   float("nan")), 3),
        }
        if gt_depth_global is not None and t == 0:
            per_frame["abs_rel_gt"] = round(abs_rel(depth[t], gt_depth_global), 4)

        all_stats.append(per_frame)

        # ── Additional threshold comparison (printed, not drawn) ──────────────
        if len(thresholds) > 1:
            thr_summary = {
                f"dyn%_thr={thr:.2f}": round(float(dyn_masks[thr][t].mean()) * 100, 1)
                for thr in thresholds
            }
            per_frame["threshold_sweep"] = thr_summary

        # ── Build panel ───────────────────────────────────────────────────────
        panel = make_panel(
            rgb_np       = images_np[t],
            depth_np     = depth[t],
            depth_conf_np= depth_conf[t],
            raft_flow_np = rf,
            ego_flow_np  = ef,
            residual_np  = rs,
            dyn_mask_np  = dm,
            frame_idx    = t,
            stats        = {
                "dyn%": frac_dyn,
                "r_med": r_stats.get("median", float("nan")),
                "r_p95": r_stats.get("p95",   float("nan")),
            },
            max_flow_mag = max_mag,
        )

        out_path = os.path.join(args.save_dir, "panels", f"{t:04d}.png")
        cv2.imwrite(out_path, panel)
        panel_paths.append(out_path)

        if t % 10 == 0 or t == S - 1:
            print(f"  frame {t:3d}/{S-1}  dyn={frac_dyn:5.1f}%  "
                  f"resid_median={r_stats.get('median', float('nan')):.2f}  "
                  f"raft_mean={raft_mag:.2f}  ego_mean={ego_mag:.2f}")

    # ── 11. Save stats JSON ────────────────────────────────────────────────────
    stats_path = os.path.join(args.save_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"[inspect] stats saved → {stats_path}")

    # ── 12. Summary across all frames ─────────────────────────────────────────
    dyn_fracs    = [s["dyn_%"]       for s in all_stats]
    resid_meds   = [s["resid_median"] for s in all_stats if not np.isnan(s["resid_median"])]
    print("\n── Summary ────────────────────────────────────────────────────────")
    print(f"  Frames analysed : {S}")
    print(f"  Dyn pixel% mean : {np.mean(dyn_fracs):.1f}%  "
          f"(min {np.min(dyn_fracs):.1f}%  max {np.max(dyn_fracs):.1f}%)")
    print(f"  Residual median avg: {np.mean(resid_meds):.2f} px")
    if gt_depth_global is not None:
        ar_vals = [s.get("abs_rel_gt") for s in all_stats if s.get("abs_rel_gt")]
        if ar_vals:
            print(f"  Abs Rel (frame 0 vs GT): {ar_vals[0]:.4f}")
    print(f"  Threshold sweep  : {thresholds}")
    for thr in thresholds:
        fracs = [float(dyn_masks[thr][t].mean()) * 100 for t in range(S)]
        print(f"    thr={thr:.2f} → dyn% mean={np.mean(fracs):.1f}%")
    print("────────────────────────────────────────────────────────────────────")

    # ── 13. Optional GIF ──────────────────────────────────────────────────────
    if args.gif:
        try:
            import imageio.v2 as iio
            frames_rgb = [
                cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                for p in panel_paths
            ]
            gif_path = os.path.join(args.save_dir, "viz.gif")
            iio.mimwrite(gif_path, frames_rgb, fps=5, loop=0)
            print(f"[inspect] GIF saved → {gif_path}")
        except Exception as e:
            print(f"[inspect] GIF failed (skipping): {e}")

    print(f"\n[inspect] done.  panels → {args.save_dir}/panels/")


if __name__ == "__main__":
    main()
