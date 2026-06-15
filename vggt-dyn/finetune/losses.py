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


def camera_loss_single(pred_pose_enc: torch.Tensor, gt_pose_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split L1 pose-encoding loss into translation / rotation(quat) / focal(FoV)
    components, following VGGT's "absT_quaR_FoV" pose encoding layout."""
    loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
    loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
    loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()

    loss_T = check_and_fix_inf_nan(loss_T).clamp(max=100).mean()
    loss_R = check_and_fix_inf_nan(loss_R).mean()
    loss_FL = check_and_fix_inf_nan(loss_FL).mean()
    return loss_T, loss_R, loss_FL


def _compute_camera_loss(
    pose_enc_list,                 # list of [1, S, D] predicted pose encodings (one per stage)
    gt_enc: torch.Tensor,          # [S, D]
    valid_frame_mask: torch.Tensor,  # [S] bool
    gamma: float = 0.6,
    weight_trans: float = 1.0,
    weight_rot: float = 1.0,
    weight_focal: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-stage camera loss with temporal decay weighting (later stages
    weighted more heavily) and separate T/R/FoV component weights."""
    n_stages = len(pose_enc_list)
    device = gt_enc.device

    if valid_frame_mask.sum() == 0:
        zero = (pose_enc_list[-1] * 0).mean()
        return zero, zero, zero, zero

    gt = gt_enc[valid_frame_mask]
    total_T = total_R = total_FL = torch.tensor(0.0, device=device)
    for stage_idx, pred_enc in enumerate(pose_enc_list):
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred = pred_enc
        if pred.dim() == 3:  # [1, S, D] -> [S, D]
            pred = pred.squeeze(0)
        loss_T, loss_R, loss_FL = camera_loss_single(pred[valid_frame_mask], gt)
        total_T = total_T + loss_T * stage_weight
        total_R = total_R + loss_R * stage_weight
        total_FL = total_FL + loss_FL * stage_weight

    avg_T = total_T / n_stages
    avg_R = total_R / n_stages
    avg_FL = total_FL / n_stages
    loss_camera = avg_T * weight_trans + avg_R * weight_rot + avg_FL * weight_focal
    return loss_camera, avg_T, avg_R, avg_FL


def gradient_loss(
    pred: torch.Tensor,    # [S, H, W, C]
    gt: torch.Tensor,      # [S, H, W, C]
    mask: torch.Tensor,    # [S, H, W] bool
    conf: torch.Tensor = None,  # [S, H, W]
    conf_alpha: float = 0.2,
    scales: int = 3,
) -> torch.Tensor:
    """Multi-scale L1 loss between x/y spatial gradients of the (pred-gt)
    error map (MonST3R-style "grad" loss) -- encourages locally consistent
    structure rather than just per-pixel accuracy."""
    device = pred.device
    total = torch.tensor(0.0, device=device)
    n_valid_scales = 0

    for scale in range(scales):
        step = 2 ** scale
        p = pred[:, ::step, ::step]
        g = gt[:, ::step, ::step]
        m = mask[:, ::step, ::step]
        c = conf[:, ::step, ::step] if conf is not None else None

        if m.shape[1] < 2 or m.shape[2] < 2:
            continue

        diff = (p - g) * m[..., None]

        grad_x = (diff[:, :, 1:] - diff[:, :, :-1]).abs().clamp(max=100)
        mask_x = m[:, :, 1:] & m[:, :, :-1]
        grad_x = grad_x * mask_x[..., None]

        grad_y = (diff[:, 1:, :] - diff[:, :-1, :]).abs().clamp(max=100)
        mask_y = m[:, 1:, :] & m[:, :-1, :]
        grad_y = grad_y * mask_y[..., None]

        if c is not None:
            conf_x = c[:, :, 1:, None].clamp(min=1e-6)
            conf_y = c[:, 1:, :, None].clamp(min=1e-6)
            grad_x = (grad_x * conf_x - conf_alpha * conf_x.log()) * mask_x[..., None]
            grad_y = (grad_y * conf_y - conf_alpha * conf_y.log()) * mask_y[..., None]

        n_valid = mask_x.sum() + mask_y.sum()
        if n_valid == 0:
            continue

        total = total + (grad_x.sum() + grad_y.sum()) / n_valid.clamp(min=1)
        n_valid_scales += 1

    if n_valid_scales == 0:
        return torch.tensor(0.0, device=device)
    return check_and_fix_inf_nan(total / n_valid_scales)


def monst3r_style_loss(
    predictions: Dict[str, torch.Tensor],
    gt_depth: torch.Tensor,
    gt_extrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
    conf_alpha_point: float = 0.2,
    conf_alpha_depth: float = 0.2,
    point_weight: float = 1.0,
    depth_weight: float = 1.0,
    camera_weight: float = 0.5,
    camera_gamma: float = 0.6,
    weight_trans: float = 1.0,
    weight_rot: float = 1.0,
    weight_focal: float = 0.5,
    gradient_loss_weight: float = 0.0,
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

    loss_grad_point = torch.tensor(0.0, device=device)
    if valid.sum() > 0:
        point_err = (pred_pts[valid] - gt_pts[valid]).norm(dim=-1)
        conf = pred_conf[valid]
        point_loss_elems = point_err * conf - conf_alpha_point * conf.log()
        point_loss_elems = check_and_fix_inf_nan(point_loss_elems)
        loss_point = filter_by_quantile(point_loss_elems, valid_range).mean()

        if gradient_loss_weight > 0:
            loss_grad_point = gradient_loss(pred_pts, gt_pts, valid, conf=pred_conf, conf_alpha=conf_alpha_point)
    else:
        loss_point = torch.tensor(0.0, device=device)

    pred_depth = predictions["depth"].squeeze(-1)
    pred_depth_conf = predictions["depth_conf"].clamp(min=1e-6)
    if pred_depth.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_depth = pred_depth.squeeze(0)
    if pred_depth_conf.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_depth_conf = pred_depth_conf.squeeze(0)

    loss_grad_depth = torch.tensor(0.0, device=device)
    if valid.sum() > 0:
        depth_err = (pred_depth[valid] - gt_depth_n[valid]).abs()
        conf_d = pred_depth_conf[valid]
        depth_loss_elems = depth_err * conf_d - conf_alpha_depth * conf_d.log()
        depth_loss_elems = check_and_fix_inf_nan(depth_loss_elems)
        loss_depth = filter_by_quantile(depth_loss_elems, valid_range).mean()

        if gradient_loss_weight > 0:
            loss_grad_depth = gradient_loss(
                pred_depth.unsqueeze(-1), gt_depth_n.unsqueeze(-1), valid,
                conf=pred_depth_conf, conf_alpha=conf_alpha_depth,
            )
    else:
        loss_depth = torch.tensor(0.0, device=device)

    loss_camera = avg_T = avg_R = avg_FL = torch.tensor(0.0, device=device)
    if "pose_enc_list" in predictions:
        gt_enc = extri_intri_to_pose_encoding(
            gt_extrinsics_n.unsqueeze(0),
            gt_intrinsics.unsqueeze(0),
            image_size_hw=(H, W),
        )
        gt_enc = check_and_fix_inf_nan(gt_enc, hard_max=None).squeeze(0)  # [S, D]

        # Only supervise camera pose on frames with enough valid GT depth.
        valid_frame_mask = valid.sum(dim=[-2, -1]) > 100

        loss_camera, avg_T, avg_R, avg_FL = _compute_camera_loss(
            predictions["pose_enc_list"], gt_enc, valid_frame_mask,
            gamma=camera_gamma, weight_trans=weight_trans, weight_rot=weight_rot, weight_focal=weight_focal,
        )

    loss_point = check_and_fix_inf_nan(loss_point + gradient_loss_weight * loss_grad_point) * point_weight
    loss_depth = check_and_fix_inf_nan(loss_depth + gradient_loss_weight * loss_grad_depth) * depth_weight
    loss_camera = check_and_fix_inf_nan(loss_camera)

    total = loss_point + loss_depth + camera_weight * loss_camera
    total = check_and_fix_inf_nan(total)
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_point": float(loss_point.detach().cpu()),
        "loss_depth": float(loss_depth.detach().cpu()),
        "loss_camera": float(loss_camera.detach().cpu()),
        "loss_T": float(avg_T.detach().cpu()),
        "loss_R": float(avg_R.detach().cpu()),
        "loss_FL": float(avg_FL.detach().cpu()),
        "loss_grad_point": float(loss_grad_point.detach().cpu()),
        "loss_grad_depth": float(loss_grad_depth.detach().cpu()),
    }
