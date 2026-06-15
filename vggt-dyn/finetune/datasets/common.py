import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


# Two fixed output buckets (H, W), both multiples of 14 (VGGT patch size).
# Datasets are assigned to whichever bucket is closest to their native
# (post principal-point-crop) aspect ratio, to minimize field-of-view loss.
DATASET_TARGET_HW = {
    "point_odyssey": (294, 518),  # ~16:9 (matches 960x540)
    "spring":        (294, 518),  # ~16:9 (matches 1920x1080)
    "tartanair":     (392, 518),  # ~4:3  (matches 640x480)
    "waymo":         (392, 518),  # ~1.5  (matches 512x341, closer to 4:3 than 16:9)
    "sintel":        (224, 518),  # ~2.35:1 (matches 1024x436)
}


def load_rgb(path: str) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def load_rgb_np(path: str) -> np.ndarray:
    """Load RGB image as HxWx3 float32 numpy array in [0, 1]."""
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def sliding_windows(n: int, clip_len: int, stride: int) -> List[List[int]]:
    windows = []
    if n < 2:
        return windows
    step = max(1, stride)
    for start in range(0, max(1, n - clip_len + 1), step):
        end = min(start + clip_len, n)
        if end - start >= 2:
            windows.append(list(range(start, end)))
    return windows


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(c2w)[:3, :].astype(np.float32)


def make_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def must_exist(path: str, msg: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg}: {path}")


def make_clip_aug_params(aug_crop: int, rng: Optional[np.random.Generator] = None) -> dict:
    """Sample one set of (pad, crop-position) params shared by all frames in a clip.

    pad: extra pixels (0..aug_crop) added to the resize target before the final crop.
    ox, oy: fractional position (0..1) of the final crop window when pad > 0
            (0.5 == centered). Ignored when pad == 0 (always centered).
    """
    rng = rng or np.random.default_rng()
    pad = int(rng.integers(0, aug_crop + 1)) if aug_crop > 0 else 0
    ox = float(rng.random())
    oy = float(rng.random())
    return {"pad": pad, "ox": ox, "oy": oy}


def crop_resize_to_fixed(
    img: np.ndarray,    # HxWx3 float32
    depth: np.ndarray,  # HxW float32
    K: np.ndarray,      # 3x3 float32
    target_hw: Tuple[int, int],
    aug: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop+resize a single frame to a fixed (target_H, target_W) output.

    Three steps (mirrors MonST3R's _crop_resize_if_necessary):
      1. Crop symmetrically around the principal point (cx, cy) so the
         principal point ends up at the crop center.
      2. Pick the target orientation matching the cropped aspect ratio, then
         uniformly (Lanczos / nearest) resize so both dims are >= target
         (+ optional aug_crop padding).
      3. Crop to the exact (target_H, target_W) — centered, or randomly
         positioned within the padding when aug_crop > 0.

    Intrinsics K are updated at every step so the output remains a valid
    pinhole camera matrix for the returned image/depth.
    """
    aug = aug or {"pad": 0, "ox": 0.5, "oy": 0.5}
    K = K.copy()
    H, W = depth.shape

    # --- Step 1: crop around principal point ---
    cx, cy = float(K[0, 2]), float(K[1, 2])
    mx = min(cx, W - cx)
    my = min(cy, H - cy)
    if mx > 1 and my > 1:
        x0, x1 = int(round(cx - mx)), int(round(cx + mx))
        y0, y1 = int(round(cy - my)), int(round(cy + my))
        img = img[y0:y1, x0:x1]
        depth = depth[y0:y1, x0:x1]
        K[0, 2] -= x0
        K[1, 2] -= y0

    H1, W1 = depth.shape
    target_H, target_W = target_hw

    # --- Step 2: orientation match + uniform resize ---
    if (H1 > W1) != (target_H > target_W):
        target_H, target_W = target_W, target_H

    pad = aug["pad"]
    goal_H, goal_W = target_H + pad, target_W + pad
    scale = max(goal_H / H1, goal_W / W1)
    new_H = max(goal_H, int(round(H1 * scale)))
    new_W = max(goal_W, int(round(W1 * scale)))
    scale_y, scale_x = new_H / H1, new_W / W1

    img = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LANCZOS4)
    depth = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y

    # --- Step 3: final crop to exact (target_H, target_W) ---
    max_x0, max_y0 = new_W - target_W, new_H - target_H
    if pad > 0:
        x0 = int(round(aug["ox"] * max_x0))
        y0 = int(round(aug["oy"] * max_y0))
    else:
        x0, y0 = max_x0 // 2, max_y0 // 2
    img = img[y0:y0 + target_H, x0:x0 + target_W]
    depth = depth[y0:y0 + target_H, x0:x0 + target_W]
    K[0, 2] -= x0
    K[1, 2] -= y0

    return img, depth, K
