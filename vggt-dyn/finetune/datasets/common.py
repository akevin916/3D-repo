import os
from typing import List

import numpy as np
import torch
from PIL import Image


def load_rgb(path: str) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


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
