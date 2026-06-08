"""
flow_utils.py — RAFT optical flow for adjacent frame pairs.

Adapted from monst3r/dust3r/cloud_opt/optimizer.py `get_flow()`.
Key difference: only computes adjacent (t, t+1) pairs instead of
the full pair graph used by MonST3R.
"""

import os
import sys

# Add project root so `third_party.raft` resolves as a package import,
# matching MonST3R's convention. Do NOT add third_party/ directly to
# sys.path — that would make `from raft import RAFT` inside raft.py
# resolve to itself (circular import).
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_THIRD_PARTY = os.path.join(_ROOT, "third_party")

import numpy as np
import torch
from tqdm import tqdm

from third_party.raft import load_RAFT, get_input_padder   # loaded as third_party.raft, not raft
from third_party.goem_opt import OccMask


def compute_adjacent_flow(
    images: list,
    model_path: str = None,
    chunk_size: int = 8,
    device: str = None,
):
    """Compute RAFT optical flow for adjacent frame pairs (t ↔ t+1).

    Args:
        images    : list of S numpy arrays [H, W, 3], float32, range [0, 1]
        model_path: path to RAFT .pth weights; None → uses default sintel model
        chunk_size: number of pairs to process per GPU batch
        device    : 'cuda' or 'cpu'; auto-detected if None

    Returns:
        flow_fwd  : [S-1, 2, H, W] tensor — forward flow (t → t+1)
        flow_bwd  : [S-1, 2, H, W] tensor — backward flow (t+1 → t)
        valid_fwd : [S-1, 1, H, W] bool tensor — occlusion-aware valid mask (fwd)
        valid_bwd : [S-1, 1, H, W] bool tensor — occlusion-aware valid mask (bwd)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    S = len(images)
    assert S >= 2, f"Need at least 2 frames, got {S}"

    imgs_src = np.stack(images[:-1])  # [S-1, H, W, 3]
    imgs_tgt = np.stack(images[1:])   # [S-1, H, W, 3]
    num_pairs = S - 1

    flow_net = load_RAFT(model_path)
    flow_net = flow_net.to(device).eval()
    get_valid_mask = OccMask(th=3.0)

    flow_fwd_list, flow_bwd_list = [], []

    with torch.no_grad():
        for i in tqdm(range(0, num_pairs, chunk_size), desc="Computing RAFT flow"):
            end = min(i + chunk_size, num_pairs)

            src = torch.tensor(imgs_src[i:end]).float().to(device)  # [B, H, W, 3]
            tgt = torch.tensor(imgs_tgt[i:end]).float().to(device)

            # RAFT expects [B, 3, H, W] in range [0, 255]
            src_t = src.permute(0, 3, 1, 2) * 255.0
            tgt_t = tgt.permute(0, 3, 1, 2) * 255.0

            # Pad to multiples of 8 (RAFT requirement)
            padder = get_input_padder(src_t.shape)
            src_t, tgt_t = padder.pad(src_t, tgt_t)

            flow_fwd_pad = flow_net(src_t, tgt_t, iters=20, test_mode=True)[1]
            flow_bwd_pad = flow_net(tgt_t, src_t, iters=20, test_mode=True)[1]

            flow_fwd_list.append(padder.unpad(flow_fwd_pad))
            flow_bwd_list.append(padder.unpad(flow_bwd_pad))

    flow_fwd = torch.cat(flow_fwd_list, dim=0)  # [S-1, 2, H, W]
    flow_bwd = torch.cat(flow_bwd_list, dim=0)

    valid_fwd = get_valid_mask(flow_fwd, flow_bwd)  # [S-1, 1, H, W]
    valid_bwd = get_valid_mask(flow_bwd, flow_fwd)

    del flow_net
    torch.cuda.empty_cache()

    return flow_fwd, flow_bwd, valid_fwd, valid_bwd
