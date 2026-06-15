"""dynamic_mask.py — Dynamic-object mask computation for vggt-dyn.

Pipeline (called every N optimisation iterations):
  1. compute_ego_flow        — ego-motion flow from current depth + pose
  2. compute_flow_residual   — compare ego_flow vs RAFT flow → residual map
  3. build_dynamic_mask      — threshold residual → bool mask [S-1, 1, H, W]
  4. pair_mask_to_frame_mask — convert per-pair mask to per-frame mask [S, 1, H, W]
  5. frame_mask_to_pair_mask — inverse of above
  6. refine_with_sam2        — optional SAM2 temporal refinement (per-frame)
  7. get_dynamic_mask_from_optimizer — one-call convenience wrapper

Reference: MonST3R get_motion_mask_from_pairs / refine_motion_mask_w_sam2.
"""

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from third_party.goem_opt import DepthBasedWarping

# ---------------------------------------------------------------------------
# Module-level DepthBasedWarping singleton (no parameters — cheap to reuse)
# ---------------------------------------------------------------------------

_warper: DepthBasedWarping = DepthBasedWarping()


def _get_warper(device: torch.device) -> DepthBasedWarping:
    global _warper
    _warper = _warper.to(device)
    return _warper


# ---------------------------------------------------------------------------
# 1. Ego-flow computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_ego_flow(
    depth: torch.Tensor,  # [S, H, W]
    R: torch.Tensor,      # [S, 3, 3]  cam-from-world
    T: torch.Tensor,      # [S, 3]
    K: torch.Tensor,      # [S, 3, 3]
) -> torch.Tensor:
    """Compute adjacent-pair ego-motion optical flow from depth and pose.

    For each pair (t, t+1) the depth of frame t is projected into frame t+1's
    image plane using DepthBasedWarping, giving the expected pixel displacement
    if the scene were completely static.

    Returns:
        ego_flow : [S-1, 2, H, W]  float32, flow (x, y) in pixels
    """
    S = depth.shape[0]
    H, W = depth.shape[1], depth.shape[2]

    if S < 2:
        return torch.zeros(0, 2, H, W, device=depth.device, dtype=depth.dtype)

    warper = _get_warper(depth.device)

    K_src = K[:-1]                     # [S-1, 3, 3]
    K_inv = torch.linalg.inv(K_src)    # [S-1, 3, 3]

    # R, T are cam-from-world (X_cam = R @ X_world + T); DepthBasedWarping
    # expects cam-to-world (X_world = R_c2w @ X_cam + t_c2w).
    R_c2w = R.transpose(-1, -2)              # [S, 3, 3]
    t_c2w = -(R_c2w @ T.unsqueeze(-1))       # [S, 3, 1]

    # DepthBasedWarping expects src_t / tgt_t as column vectors [B, 3, 1]
    ego_hom, _ = warper(
        R_c2w[:-1],                    # src_R  [S-1, 3, 3]
        t_c2w[:-1],                    # src_t  [S-1, 3, 1]
        R_c2w[1:],                     # tgt_R  [S-1, 3, 3]
        t_c2w[1:],                     # tgt_t  [S-1, 3, 1]
        depth[:-1].unsqueeze(1),       # src_depth [S-1, 1, H, W]
        K_src,
        K_inv,
        use_depth=True,                # metric depth (not disparity)
    )
    # ego_hom: [S-1, 3, H, W]; channels 0-1 are pixel-space flow (x, y)
    return ego_hom[:, :2].contiguous()  # [S-1, 2, H, W]


# ---------------------------------------------------------------------------
# 2. Flow residual
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_flow_residual(
    ego_flow:  torch.Tensor,   # [S-1, 2, H, W]
    raft_flow: torch.Tensor,   # [S-1, 2, H, W]
    valid_mask: torch.Tensor,  # [S-1, 1, H, W] bool — valid RAFT pixels
) -> torch.Tensor:
    """Per-pixel L2 norm of (ego_flow − raft_flow).

    Pixels flagged as invalid (occluded / out-of-bounds by RAFT) are set to 0
    so they do not inflate the per-pair statistics used for normalisation.

    Returns:
        residual : [S-1, 1, H, W]  float32
    """
    diff     = ego_flow - raft_flow                # [S-1, 2, H, W]
    mag      = diff.norm(dim=1, keepdim=True)      # [S-1, 1, H, W]
    residual = mag * valid_mask.float()            # zero out invalid pixels
    return residual


# ---------------------------------------------------------------------------
# 3. Build dynamic mask
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_dynamic_mask(
    flow_residual: torch.Tensor,  # [S-1, 1, H, W]
    threshold: float = 0.35,
    normalize: bool  = True,
) -> torch.Tensor:
    """Threshold flow residual → binary dynamic-pixel mask.

    Args:
        flow_residual : [S-1, 1, H, W] residual magnitudes
        threshold     : decision boundary.
                        • If normalize=True : fraction in [0, 1] after per-pair
                          min-max normalisation (MonST3R default: 0.35).
                        • If normalize=False: absolute pixel-unit threshold.
        normalize     : whether to normalise each pair's residual to [0, 1]
                        before thresholding (recommended; follows MonST3R).

    Returns:
        mask : [S-1, 1, H, W]  bool  — True = dynamic pixel
    """
    if normalize:
        B     = flow_residual.shape[0]
        r     = flow_residual.reshape(B, -1)               # [B, H*W]
        r_min = r.amin(dim=1, keepdim=True)
        r_max = r.amax(dim=1, keepdim=True)
        denom = (r_max - r_min).clamp(min=1e-6)
        r_norm = (r - r_min) / denom                       # [B, H*W] in [0,1]
        residual_for_thresh = r_norm.reshape_as(flow_residual)
    else:
        residual_for_thresh = flow_residual

    return residual_for_thresh > threshold


# ---------------------------------------------------------------------------
# 4. Per-pair ↔ per-frame conversion helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def pair_mask_to_frame_mask(
    pair_mask: torch.Tensor,  # [S-1, 1, H, W] bool
    S: int,
) -> torch.Tensor:
    """Aggregate per-pair masks → per-frame masks via OR.

    Frame t is flagged dynamic if any pair containing frame t has that pixel
    marked as dynamic:  frame t ← pair(t-1, t) | pair(t, t+1).

    Returns:
        frame_mask : [S, 1, H, W]  bool
    """
    _, _, H, W = pair_mask.shape
    device = pair_mask.device
    frame_mask = torch.zeros(S, 1, H, W, dtype=torch.bool, device=device)
    for k in range(S - 1):
        frame_mask[k]     = frame_mask[k]   | pair_mask[k]
        frame_mask[k + 1] = frame_mask[k + 1] | pair_mask[k]
    return frame_mask


@torch.no_grad()
def frame_mask_to_pair_mask(
    frame_mask: torch.Tensor,  # [S, 1, H, W] bool
) -> torch.Tensor:
    """Convert per-frame masks → per-pair masks via OR of adjacent frames.

    Returns:
        pair_mask : [S-1, 1, H, W]  bool
    """
    return frame_mask[:-1] | frame_mask[1:]


# ---------------------------------------------------------------------------
# 5. SAM2 temporal refinement (optional)
# ---------------------------------------------------------------------------

def refine_with_sam2(
    frame_mask: torch.Tensor,  # [S, 1, H, W] bool, per-frame
    images: torch.Tensor,      # [S, H, W, 3] float32 in [0,1] or uint8
    sam2_checkpoint: str = "third_party/sam2/checkpoints/sam2.1_hiera_large.pt",
    model_cfg: str       = "configs/sam2.1/sam2.1_hiera_l.yaml",
) -> torch.Tensor:
    """Optional SAM2-based temporal mask refinement.

    Mirrors MonST3R's ``refine_motion_mask_w_sam2``:
      • Propagates the flow-based mask through the video using SAM2's video
        predictor with two alternating passes (odd-frame seeds, even-frame seeds).
      • ORs both passes and the original flow mask for robustness.

    Returns:
        refined_mask : [S, 1, H, W]  bool
    Raises:
        ImportError      : if ``sam2`` package is not installed.
        FileNotFoundError: if ``sam2_checkpoint`` does not exist.
    """
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise ImportError(
            "SAM2 is not installed. "
            "Build it from third_party/sam2 or skip SAM2 refinement."
        ) from exc

    if not os.path.isfile(sam2_checkpoint):
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {sam2_checkpoint}. "
            "Download it or set sam2_checkpoint to a valid path."
        )

    device = frame_mask.device
    device_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
    S = frame_mask.shape[0]

    # Normalise images to [0, 1] float32 and transpose to [S, 3, H, W]
    if images.dtype == torch.uint8:
        frames_t = images.permute(0, 3, 1, 2).float() / 255.0
    else:
        frames_t = images.permute(0, 3, 1, 2).float()
    frames_t = frames_t.to(device)

    mask_list = [frame_mask[i, 0].bool() for i in range(S)]
    sam2_masks = [None] * S

    # Enable TF32 for Ampere+ GPUs (follows MonST3R)
    prev_tf32, prev_cudnn = False, False
    if device_type == "cuda":
        prev_tf32  = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn = torch.backends.cudnn.allow_tf32
        if torch.cuda.get_device_properties(device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True

    autocast_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
    ann_obj_id = 1

    try:
        with torch.autocast(device_type=device_type, dtype=autocast_dtype):
            predictor      = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=str(device))
            inference_state = predictor.init_state(video_path=frames_t)

            # Two passes: odd-indexed frames as seeds (parity=1) then even (parity=0).
            # Each pass fills in the *other* parity from video propagation.
            for seed_parity in (1, 0):
                predictor.reset_state(inference_state)
                for idx in range(S):
                    if idx % 2 == seed_parity:
                        predictor.add_new_mask(
                            inference_state,
                            frame_idx=idx,
                            obj_id=ann_obj_id,
                            mask=mask_list[idx].to(device),
                        )

                video_segments = {}
                for out_idx, out_obj_ids, out_logits in predictor.propagate_in_video(
                    inference_state, start_frame_idx=0
                ):
                    video_segments[out_idx] = {
                        oid: (out_logits[i] > 0.0).cpu()
                        for i, oid in enumerate(out_obj_ids)
                    }

                # Store propagated result for frames *not* used as seeds
                fill_parity = 1 - seed_parity
                for idx in range(S):
                    if idx % 2 == fill_parity:
                        raw = video_segments[idx][ann_obj_id]
                        m   = raw[0].bool() if isinstance(raw, torch.Tensor) else __import__("torch").from_numpy(raw[0]).bool()
                        sam2_masks[idx] = m if sam2_masks[idx] is None else (sam2_masks[idx] | m)

            del predictor

    finally:
        if device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
            torch.backends.cudnn.allow_tf32        = prev_cudnn

    # OR sam2 result with original flow mask (conservative union)
    result = []
    for i in range(S):
        base  = mask_list[i].cpu()
        extra = sam2_masks[i] if sam2_masks[i] is not None else torch.zeros_like(base)
        result.append((base | extra).unsqueeze(0))   # [1, H, W]

    return torch.stack(result, dim=0).to(device)      # [S, 1, H, W]


# ---------------------------------------------------------------------------
# 6. Convenience wrapper: derive mask from optimizer's current state
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_dynamic_mask_from_optimizer(
    optimizer,                    # VGGTDynOptimizer
    threshold: float = 0.35,
    normalize: bool  = True,
) -> torch.Tensor:
    """Compute a fresh [S-1, 1, H, W] dynamic mask from the optimizer's
    current refined depth and pose estimates.

    Call this every N iterations inside the optimisation loop and pass the
    result to ``optimizer.update_dynamic_mask(mask)``:

    Example::

        for it in range(niter):
            optimization_step(net, it, niter, lr, lr_min, adam)
            if it % 10 == 9:
                mask = get_dynamic_mask_from_optimizer(net)
                net.update_dynamic_mask(mask)

    Args:
        optimizer : VGGTDynOptimizer instance.
        threshold : dynamic-pixel threshold (see ``build_dynamic_mask``).
        normalize : whether to normalise residual per pair (recommended).

    Returns:
        mask : [S-1, 1, H, W]  bool — True = dynamic pixel
    """
    depth    = optimizer.get_refined_depth()   # [S, H, W]
    R, T     = optimizer.get_refined_RT()      # [S,3,3], [S,3]
    K        = optimizer.K                     # [S, 3, 3]

    ego_flow = compute_ego_flow(depth, R, T, K)
    residual = compute_flow_residual(
        ego_flow,
        optimizer.flow_fwd,
        optimizer.valid_fwd,
    )
    return build_dynamic_mask(residual, threshold=threshold, normalize=normalize)
