#!/usr/bin/env python3
"""Sintel pose evaluation aligned with MonST3R-style protocol.

Inputs:
  - predicted camera extrinsics from run.py output:
      <output_dir>/extrinsics.npy   shape [S, 3, 4], cam-from-world (world->cam)
  - Sintel GT camera files:
      <gt_cam_dir>/frame_XXXX.cam   each contains intrinsics + extrinsics (world->cam)

Metrics:
  - ATE RMSE on translation after global Sim(3) alignment
  - RPE translation RMSE (delta=1 frame)
  - RPE rotation RMSE in degrees (delta=1 frame)

This script saves:
  - <output_dir>/pred_traj.txt
  - <output_dir>/pred_traj_aligned.txt
  - <output_dir>/gt_traj.txt
  - <output_dir>/pose_eval_metric.txt
  - <output_dir>/eval_sintel_pose.json
"""

import os
import re
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

TAG_FLOAT = 202021.25


def _sintel_cam_read(path: str):
    with open(path, "rb") as f:
        tag = np.fromfile(f, dtype=np.float32, count=1)[0]
        if abs(float(tag) - TAG_FLOAT) > 1e-4:
            raise ValueError(f"Wrong Sintel .cam tag in {path}: {tag}")
        intrinsics = np.fromfile(f, dtype=np.float64, count=9).reshape((3, 3))
        extrinsics = np.fromfile(f, dtype=np.float64, count=12).reshape((3, 4))
    return intrinsics, extrinsics


def _extract_frame_id(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"(\d+)", name)
    if m is None:
        raise ValueError(f"Cannot parse frame id from {name}")
    return int(m.group(1))


def _w2c34_to_c2w44(w2c: np.ndarray) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :4] = w2c
    return np.linalg.inv(m)


def _poses_to_tum(poses_c2w: np.ndarray) -> np.ndarray:
    xyz = poses_c2w[:, :3, 3]
    quat_xyzw = R.from_matrix(poses_c2w[:, :3, :3]).as_quat()
    quat_wxyz = np.stack(
        [quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]],
        axis=1,
    )
    return np.concatenate([xyz, quat_wxyz], axis=1)


def _save_tum(traj_tum: np.ndarray, timestamps: np.ndarray, path: str):
    with open(path, "w") as f:
        for i in range(traj_tum.shape[0]):
            vals = " ".join(f"{x:.12f}" for x in traj_tum[i])
            f.write(f"{timestamps[i]:.0f} {vals}\n")


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Compute Sim(3) such that dst ~= s * R * src + t."""
    assert src.shape == dst.shape and src.shape[1] == 3
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


def _apply_sim3_to_poses(poses_c2w: np.ndarray, scale: float, r_sim: np.ndarray, t_sim: np.ndarray):
    out = poses_c2w.copy()
    xyz = out[:, :3, 3]
    xyz_aligned = (scale * (r_sim @ xyz.T)).T + t_sim
    out[:, :3, 3] = xyz_aligned
    out[:, :3, :3] = np.einsum("ij,njk->nik", r_sim, out[:, :3, :3])
    return out


def _relative_pose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.inv(a) @ b


def _rotation_angle_deg(rot_mat: np.ndarray) -> float:
    angle = np.arccos(np.clip((np.trace(rot_mat) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(angle))


def _compute_metrics(gt_c2w: np.ndarray, pred_c2w: np.ndarray):
    if gt_c2w.shape[0] != pred_c2w.shape[0]:
        raise ValueError(f"Length mismatch: gt={gt_c2w.shape[0]} pred={pred_c2w.shape[0]}")
    if gt_c2w.shape[0] < 2:
        raise ValueError("Need at least 2 poses for RPE")

    gt_xyz = gt_c2w[:, :3, 3]
    pred_xyz = pred_c2w[:, :3, 3]

    scale, r_sim, t_sim = _umeyama_sim3(pred_xyz, gt_xyz)
    pred_aligned = _apply_sim3_to_poses(pred_c2w, scale, r_sim, t_sim)

    ate = float(np.sqrt(np.mean(np.sum((pred_aligned[:, :3, 3] - gt_xyz) ** 2, axis=1))))

    trans_err = []
    rot_err = []
    for i in range(gt_c2w.shape[0] - 1):
        j = i + 1
        gt_rel = _relative_pose(gt_c2w[i], gt_c2w[j])
        pr_rel = _relative_pose(pred_aligned[i], pred_aligned[j])
        err = np.linalg.inv(gt_rel) @ pr_rel
        trans_err.append(np.linalg.norm(err[:3, 3]))
        rot_err.append(_rotation_angle_deg(err[:3, :3]))

    rpe_t = float(np.sqrt(np.mean(np.square(trans_err))))
    rpe_r = float(np.sqrt(np.mean(np.square(rot_err))))

    return {
        "ate": ate,
        "rpe_trans": rpe_t,
        "rpe_rot": rpe_r,
        "scale": float(scale),
    }, pred_aligned


def _load_gt_sintel_traj(gt_cam_dir: str, stride: int):
    cam_files = sorted(
        [os.path.join(gt_cam_dir, f) for f in os.listdir(gt_cam_dir) if f.endswith(".cam")]
    )
    if not cam_files:
        raise FileNotFoundError(f"No .cam files found in {gt_cam_dir}")

    cam_files = cam_files[::stride]

    poses = []
    ts = []
    for p in cam_files:
        _, w2c = _sintel_cam_read(p)
        poses.append(_w2c34_to_c2w44(w2c))
        ts.append(float(_extract_frame_id(p)))

    poses = np.stack(poses, axis=0)
    # Match MonST3R helper behavior: center GT translation before metric eval.
    poses[:, :3, 3] -= np.mean(poses[:, :3, 3], axis=0, keepdims=True)
    return poses, np.array(ts, dtype=np.float64)


def _load_pred_traj(output_dir: str, stride: int):
    ext_path = os.path.join(output_dir, "extrinsics.npy")
    if not os.path.isfile(ext_path):
        raise FileNotFoundError(f"extrinsics.npy not found in {output_dir}")

    ext = np.load(ext_path)
    if ext.ndim != 3 or ext.shape[1:] != (3, 4):
        raise ValueError(f"Unexpected extrinsics shape: {ext.shape}")

    ext = ext[::stride]
    poses = np.stack([_w2c34_to_c2w44(e) for e in ext], axis=0)
    ts = np.arange(poses.shape[0], dtype=np.float64)
    return poses, ts


def parse_args():
    p = argparse.ArgumentParser(description="Sintel pose evaluation (MonST3R-aligned metrics)")
    p.add_argument("--output_dir", required=True, help="run.py output dir containing extrinsics.npy")
    p.add_argument("--gt_cam_dir", default=None, help="Sintel camdata_left/<seq> directory")
    p.add_argument("--gt_traj_dir", default=None,
                   help="deprecated alias of --gt_cam_dir, kept for backward compatibility")
    p.add_argument("--pose_eval_stride", type=int, default=1, help="frame stride for evaluation")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.abspath(args.output_dir)
    gt_cam_dir_arg = args.gt_cam_dir or args.gt_traj_dir
    if gt_cam_dir_arg is None:
        raise ValueError("Either --gt_cam_dir or --gt_traj_dir must be provided")
    gt_cam_dir = os.path.abspath(gt_cam_dir_arg)

    gt_c2w, gt_ts = _load_gt_sintel_traj(gt_cam_dir, stride=args.pose_eval_stride)
    pred_c2w, pred_ts = _load_pred_traj(output_dir, stride=args.pose_eval_stride)

    n = min(gt_c2w.shape[0], pred_c2w.shape[0])
    if n < 2:
        raise RuntimeError(f"Not enough frames for pose eval: {n}")

    gt_c2w = gt_c2w[:n]
    pred_c2w = pred_c2w[:n]
    gt_ts = gt_ts[:n]
    pred_ts = pred_ts[:n]

    metrics, pred_aligned = _compute_metrics(gt_c2w, pred_c2w)

    pred_tum = _poses_to_tum(pred_c2w)
    gt_tum = _poses_to_tum(gt_c2w)
    pred_aligned_tum = _poses_to_tum(pred_aligned)

    os.makedirs(output_dir, exist_ok=True)
    _save_tum(pred_tum, pred_ts, os.path.join(output_dir, "pred_traj.txt"))
    _save_tum(pred_aligned_tum, gt_ts, os.path.join(output_dir, "pred_traj_aligned.txt"))
    _save_tum(gt_tum, gt_ts, os.path.join(output_dir, "gt_traj.txt"))

    metric_txt = os.path.join(output_dir, "pose_eval_metric.txt")
    with open(metric_txt, "w") as f:
        f.write("Sintel Pose Evaluation\n")
        f.write(f"Frames: {n}\n")
        f.write(f"ATE: {metrics['ate']:.6f}\n")
        f.write(f"RPE trans: {metrics['rpe_trans']:.6f}\n")
        f.write(f"RPE rot: {metrics['rpe_rot']:.6f}\n")
        f.write(f"Scale: {metrics['scale']:.6f}\n")

    result = {
        "num_frames": int(n),
        "pose_eval_stride": int(args.pose_eval_stride),
        **metrics,
    }
    result_json = os.path.join(output_dir, "eval_sintel_pose.json")
    with open(result_json, "w") as f:
        json.dump(result, f, indent=2)

    print(
        f"[eval:sintel_pose] ATE={metrics['ate']:.6f}, "
        f"RPE_t={metrics['rpe_trans']:.6f}, RPE_r={metrics['rpe_rot']:.6f}"
    )
    print(f"[eval:sintel_pose] saved -> {result_json}")


if __name__ == "__main__":
    main()
