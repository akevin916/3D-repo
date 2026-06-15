"""Pose alignment / error metrics for validation (mirrors evaluators/sintel_pose.py)."""

from typing import Tuple

import numpy as np


def w2c_to_c2w(extri_w2c: np.ndarray) -> np.ndarray:
    """Convert [S, 3, 4] world-to-cam extrinsics to [S, 4, 4] cam-to-world."""
    S = extri_w2c.shape[0]
    c2w = np.tile(np.eye(4, dtype=np.float64), (S, 1, 1))
    R = extri_w2c[:, :3, :3].astype(np.float64)
    t = extri_w2c[:, :3, 3].astype(np.float64)
    R_T = np.transpose(R, (0, 2, 1))
    c2w[:, :3, :3] = R_T
    c2w[:, :3, 3] = -np.einsum("sij,sj->si", R_T, t)
    return c2w


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Compute Sim(3) such that dst ~= s * R * src + t. (mirrors evaluators/sintel_pose.py)"""
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    cov = (dst_c.T @ src_c) / n
    u, d, vt = np.linalg.svd(cov)

    s_mat = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        s_mat[2, 2] = -1.0

    r_mat = u @ s_mat @ vt
    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    scale = float(np.trace(np.diag(d) @ s_mat) / max(var_src, 1e-12))
    trans = mu_d - scale * (r_mat @ mu_s)
    return scale, r_mat, trans


def rotation_angle_deg(rot_mat: np.ndarray) -> float:
    angle = np.arccos(np.clip((np.trace(rot_mat) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(angle))


def pose_metrics(gt_c2w: np.ndarray, pred_c2w: np.ndarray) -> Tuple[float, float]:
    """Sim(3)-aligned ATE (RMSE of position) and RPE_rot (deg), as in evaluators/sintel_pose.py."""
    gt_xyz = gt_c2w[:, :3, 3]
    pred_xyz = pred_c2w[:, :3, 3]

    scale, r_sim, t_sim = umeyama_sim3(pred_xyz, gt_xyz)
    pred_aligned = pred_c2w.copy()
    pred_aligned[:, :3, 3] = (scale * (r_sim @ pred_xyz.T)).T + t_sim
    pred_aligned[:, :3, :3] = np.einsum("ij,njk->nik", r_sim, pred_aligned[:, :3, :3])

    ate = float(np.sqrt(np.mean(np.sum((pred_aligned[:, :3, 3] - gt_xyz) ** 2, axis=1))))

    rot_err = []
    for i in range(gt_c2w.shape[0] - 1):
        j = i + 1
        gt_rel = np.linalg.inv(gt_c2w[i]) @ gt_c2w[j]
        pr_rel = np.linalg.inv(pred_aligned[i]) @ pred_aligned[j]
        err = np.linalg.inv(gt_rel) @ pr_rel
        rot_err.append(rotation_angle_deg(err[:3, :3]))

    rpe_rot = float(np.sqrt(np.mean(np.square(rot_err))))
    return ate, rpe_rot
