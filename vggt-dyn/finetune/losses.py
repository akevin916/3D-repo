"""MonST3R-style conf-weighted regression loss for VGGT fine-tuning."""

from typing import Dict, Tuple

import torch

from vggt.utils.pose_enc import extri_intri_to_pose_encoding


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


def monst3r_style_loss(
    predictions: Dict[str, torch.Tensor],
    gt_depth: torch.Tensor,
    gt_extrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
    conf_alpha: float,
    camera_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Conf-weighted regression inspired by MonST3R's ConfLoss(Regr3D)."""
    S, H, W = gt_depth.shape
    device = gt_depth.device

    valid = gt_depth > 0
    gt_pts = _compute_world_points_from_depth(gt_depth, gt_extrinsics, gt_intrinsics)

    # Per-scene scale normalization (MonST3R avg_dis norm_mode):
    # normalize by the average distance of valid GT points to origin so that
    # all datasets (indoor ~1m, outdoor ~100m) contribute equally to the loss.
    if valid.sum() > 0:
        scene_scale = gt_pts[valid].norm(dim=-1).mean().clamp(min=1e-3).detach()
        depth_scale = gt_depth[valid].mean().clamp(min=1e-3).detach()
    else:
        scene_scale = torch.tensor(1.0, device=device)
        depth_scale = torch.tensor(1.0, device=device)

    pred_pts = predictions["world_points"]
    pred_conf = predictions["world_points_conf"].clamp(min=1e-6)
    if pred_pts.dim() == 5:  # [1, S, H, W, 3] -> [S, H, W, 3]
        pred_pts = pred_pts.squeeze(0)
    if pred_conf.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_conf = pred_conf.squeeze(0)

    if valid.sum() > 0:
        point_err = ((pred_pts[valid] - gt_pts[valid]) / scene_scale).norm(dim=-1)
        conf = pred_conf[valid]
        loss_point = (point_err * conf - conf_alpha * conf.log()).mean()
    else:
        loss_point = torch.tensor(0.0, device=device)

    pred_depth = predictions["depth"].squeeze(-1)
    pred_depth_conf = predictions["depth_conf"].clamp(min=1e-6)
    if pred_depth.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_depth = pred_depth.squeeze(0)
    if pred_depth_conf.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_depth_conf = pred_depth_conf.squeeze(0)
    if valid.sum() > 0:
        depth_err = ((pred_depth[valid] - gt_depth[valid]) / depth_scale).abs()
        conf_d = pred_depth_conf[valid]
        loss_depth = (depth_err * conf_d - conf_alpha * conf_d.log()).mean()
    else:
        loss_depth = torch.tensor(0.0, device=device)

    loss_camera = torch.tensor(0.0, device=device)
    if "pose_enc_list" in predictions:
        # Normalize GT translation by the same scene_scale (avg_dis) used for
        # point/depth loss, so gt_enc's T matches the normalized scale of
        # VGGT's predicted pose_enc (otherwise datasets with large absolute
        # translations, e.g. waymo, dominate the camera loss).
        gt_extrinsics_n = gt_extrinsics.clone()
        gt_extrinsics_n[:, :, 3] = gt_extrinsics_n[:, :, 3] / scene_scale
        gt_enc = extri_intri_to_pose_encoding(
            gt_extrinsics_n.unsqueeze(0),
            gt_intrinsics.unsqueeze(0),
            image_size_hw=(H, W),
        )
        for pred_enc in predictions["pose_enc_list"]:
            loss_camera = loss_camera + (pred_enc - gt_enc).abs().mean()
        loss_camera = loss_camera / len(predictions["pose_enc_list"])

    total = loss_point + loss_depth + camera_weight * loss_camera
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_point": float(loss_point.detach().cpu()),
        "loss_depth": float(loss_depth.detach().cpu()),
        "loss_camera": float(loss_camera.detach().cpu()),
    }
