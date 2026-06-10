import os
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


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


def center_crop_to_principal_point(
    imgs: List[np.ndarray],    # list of HxWx3
    depths: List[np.ndarray],  # list of HxW
    Ks: List[np.ndarray],      # list of 3x3
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Crop each frame symmetrically around its principal point (cx, cy).

    Mirrors MonST3R's _crop_resize_if_necessary centering logic: the window is
    a rectangle of half-widths (min_margin_x, min_margin_y) centred on (cx, cy),
    ensuring the principal point ends up exactly at the image centre after crop.
    Intrinsics are updated to reflect the new origin.
    """
    out_imgs, out_depths, out_Ks = [], [], []
    for img, depth, K in zip(imgs, depths, Ks):
        H, W = depth.shape
        cx, cy = float(K[0, 2]), float(K[1, 2])
        mx = int(min(cx, W - cx))
        my = int(min(cy, H - cy))
        if mx <= 0 or my <= 0:
            out_imgs.append(img)
            out_depths.append(depth)
            out_Ks.append(K)
            continue
        x0, x1 = int(round(cx)) - mx, int(round(cx)) + mx
        y0, y1 = int(round(cy)) - my, int(round(cy)) + my
        K_new = K.copy()
        K_new[0, 2] -= x0
        K_new[1, 2] -= y0
        out_imgs.append(img[y0:y1, x0:x1])
        out_depths.append(depth[y0:y1, x0:x1])
        out_Ks.append(K_new)
    return out_imgs, out_depths, out_Ks
