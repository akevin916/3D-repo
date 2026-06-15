"""MonST3R-style conf-weighted regression loss for VGGT fine-tuning."""

from typing import Dict, Tuple

import torch

from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from vggt.utils.geometry import closed_form_inverse_se3


def _compute_world_points_from_depth(
    depth: torch.Tensor,       # [S, H, W]
    extrinsics: torch.Tensor,  # [S, 3, 4] world-to-cam
    intrinsics: torch.Tensor,  # [S, 3, 3]
) -> torch.Tensor:
    """Back-project GT depth into GT world points [S, H, W, 3]."""
    S, H, W = depth.shape
    device = depth.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(-1, 3)

    K_inv = torch.linalg.inv(intrinsics)
    rays = torch.einsum("sij,pj->spi", K_inv, grid)
    pts_c = rays * depth.reshape(S, -1, 1)

    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    R_T = R.transpose(1, 2)
    pts_w = torch.einsum("sij,spj->spi", R_T, pts_c - t.unsqueeze(1))
    return pts_w.reshape(S, H, W, 3)


def check_and_fix_inf_nan(input_tensor: torch.Tensor, hard_max: float | None = 100.0) -> torch.Tensor:
    """Replace NaN/Inf with 0 and (optionally) clamp to [-hard_max, hard_max]."""
    if input_tensor is None:
        return input_tensor
    if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor,
        )
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)
    return input_tensor


def _torch_quantile_1d(x: torch.Tensor, q: float) -> torch.Tensor:
    """Scalar quantile via kthvalue (no torch.quantile 2**24 element limit)."""
    k = round(q * (x.numel() - 1)) + 1
    return torch.kthvalue(x, k)[0]


def filter_by_quantile(loss_tensor: torch.Tensor, valid_range: float, min_elements: int = 1000, hard_max: float = 100.0) -> torch.Tensor:
    """Clamp to hard_max, then drop elements above the `valid_range` quantile
    (outlier pixels) so they don't dominate the gradient."""
    loss_tensor = loss_tensor.clamp(max=hard_max)
    if loss_tensor.numel() <= min_elements:
        return loss_tensor

    thresh = min(_torch_quantile_1d(loss_tensor.detach().reshape(-1), valid_range).item(), hard_max)
    mask = loss_tensor < thresh
    if mask.sum() > min_elements:
        return loss_tensor[mask]
    return loss_tensor


def _normalize_gt_to_cam0(
    gt_extrinsics: torch.Tensor,  # [S, 3, 4] world-to-cam
    gt_pts: torch.Tensor,         # [S, H, W, 3]
    gt_depth: torch.Tensor,       # [S, H, W]
    valid: torch.Tensor,          # [S, H, W] bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform GT extrinsics/world points into the cam0-relative frame that
    VGGT itself operates in (cam0 -> identity pose), then rescale everything
    by the avg_dis of valid points (replaces the separate
    scene_scale/depth_scale normalization with a single unified scale).
    """
    S = gt_extrinsics.shape[0]
    device, dtype = gt_extrinsics.device, gt_extrinsics.dtype

    extrinsics_homog = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(S, 1, 1)
    extrinsics_homog[:, :3, :4] = gt_extrinsics

    cam0_to_world = closed_form_inverse_se3(extrinsics_homog[:1])  # (1, 4, 4)
    new_extrinsics = torch.matmul(extrinsics_homog, cam0_to_world)[:, :3, :4]  # (S, 3, 4)

    R0 = gt_extrinsics[0, :3, :3]
    t0 = gt_extrinsics[0, :3, 3]
    new_pts = gt_pts @ R0.transpose(-1, -2) + t0

    if valid.sum() > 0:
        avg_scale = (new_pts[valid].norm(dim=-1).sum() / (valid.sum() + 1e-3)).clamp(min=1e-6, max=1e6).detach()
    else:
        avg_scale = torch.tensor(1.0, device=device, dtype=dtype)

    new_pts = new_pts / avg_scale
    new_extrinsics = new_extrinsics.clone()
    new_extrinsics[:, :3, 3] = new_extrinsics[:, :3, 3] / avg_scale
    new_depth = gt_depth / avg_scale

    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, hard_max=None)
    new_pts = check_and_fix_inf_nan(new_pts, hard_max=None)
    new_depth = check_and_fix_inf_nan(new_depth, hard_max=None)

    return new_extrinsics, new_pts, new_depth, avg_scale


def monst3r_style_loss(
    predictions: Dict[str, torch.Tensor],
    gt_depth: torch.Tensor,
    gt_extrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
    conf_alpha: float,
    camera_weight: float,
    valid_range: float = 0.98,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Conf-weighted regression inspired by MonST3R's ConfLoss(Regr3D)."""
    S, H, W = gt_depth.shape
    device = gt_depth.device

    valid = gt_depth > 0
    gt_pts_raw = _compute_world_points_from_depth(gt_depth, gt_extrinsics, gt_intrinsics)

    # Transform GT to the cam0-relative frame + avg_dis scale that VGGT's
    # predictions are already expressed in, so pred and GT are directly
    # comparable without separate per-modality scale factors.
    gt_extrinsics_n, gt_pts, gt_depth_n, avg_scale = _normalize_gt_to_cam0(
        gt_extrinsics, gt_pts_raw, gt_depth, valid
    )

    pred_pts = predictions["world_points"]
    pred_conf = predictions["world_points_conf"].clamp(min=1e-6)
    if pred_pts.dim() == 5:  # [1, S, H, W, 3] -> [S, H, W, 3]
        pred_pts = pred_pts.squeeze(0)
    if pred_conf.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_conf = pred_conf.squeeze(0)

    if valid.sum() > 0:
        point_err = (pred_pts[valid] - gt_pts[valid]).norm(dim=-1)
        conf = pred_conf[valid]
        point_loss_elems = point_err * conf - conf_alpha * conf.log()
        point_loss_elems = check_and_fix_inf_nan(point_loss_elems)
        loss_point = filter_by_quantile(point_loss_elems, valid_range).mean()
    else:
        loss_point = torch.tensor(0.0, device=device)

    pred_depth = predictions["depth"].squeeze(-1)
    pred_depth_conf = predictions["depth_conf"].clamp(min=1e-6)
    if pred_depth.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_depth = pred_depth.squeeze(0)
    if pred_depth_conf.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_depth_conf = pred_depth_conf.squeeze(0)
    if valid.sum() > 0:
        depth_err = (pred_depth[valid] - gt_depth_n[valid]).abs()
        conf_d = pred_depth_conf[valid]
        depth_loss_elems = depth_err * conf_d - conf_alpha * conf_d.log()
        depth_loss_elems = check_and_fix_inf_nan(depth_loss_elems)
        loss_depth = filter_by_quantile(depth_loss_elems, valid_range).mean()
    else:
        loss_depth = torch.tensor(0.0, device=device)

    loss_camera = torch.tensor(0.0, device=device)
    if "pose_enc_list" in predictions:
        gt_enc = extri_intri_to_pose_encoding(
            gt_extrinsics_n.unsqueeze(0),
            gt_intrinsics.unsqueeze(0),
            image_size_hw=(H, W),
        )
        gt_enc = check_and_fix_inf_nan(gt_enc, hard_max=None)
        for pred_enc in predictions["pose_enc_list"]:
            loss_camera = loss_camera + (pred_enc - gt_enc).abs().mean()
        loss_camera = loss_camera / len(predictions["pose_enc_list"])

    loss_point = check_and_fix_inf_nan(loss_point)
    loss_depth = check_and_fix_inf_nan(loss_depth)
    loss_camera = check_and_fix_inf_nan(loss_camera)

    total = loss_point + loss_depth + camera_weight * loss_camera
    total = check_and_fix_inf_nan(total)
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_point": float(loss_point.detach().cpu()),
        "loss_depth": float(loss_depth.detach().cpu()),
        "loss_camera": float(loss_camera.detach().cpu()),
    }
