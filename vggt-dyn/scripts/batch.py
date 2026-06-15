#!/usr/bin/env python3
"""Unified batch runner for VGGT-Dyn.

Subcommands
-----------
  depth        --dataset bonn|sintel|kitti
               run.py + eval.py  (depth metrics: AbsRel/RMSE/δ)

  pose
               run.py + eval.py sintel_pose  (ATE/RPE)

  single_frame --dataset bonn|sintel|kitti
               scripts/vggt_baseline.py + eval.py  (original VGGT, no TTO)

This script replaces the four root-level batch scripts:
  depth_batch.py, kitti_batch.py, sintel_pose_batch.py,
  vggt_single_frame_batch.py.
"""

import os
import re
import sys
import json
import shlex
import argparse
import subprocess
from typing import Dict, List, Optional

# ── project root (one level above scripts/) ───────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

# ── well-known sequence subsets ───────────────────────────────────────────────
MONST3R_BONN_SUBSET = [
    "balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous",
]

MONST3R_SINTEL_SUBSET = [
    "alley_2", "ambush_4", "ambush_5", "ambush_6",
    "cave_2", "cave_4",
    "market_2", "market_5", "market_6",
    "shaman_3",
    "sleeping_1", "sleeping_2",
    "temple_2", "temple_3",
]

KITTI_SEQ_RE = re.compile(r"\d{4}_\d{2}_\d{2}_drive_\d{4}_sync")


# ===========================================================================
# Shared helpers
# ===========================================================================

def _print_cmd(cmd: List[str]):
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd))


def _run_cmd(cmd: List[str], dry_run: bool) -> int:
    _print_cmd(cmd)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _parse_seq_csv(arg: str) -> List[str]:
    seqs = [s.strip() for s in (arg or "").split(",") if s.strip()]
    if not seqs:
        raise ValueError("--sequences is empty after parsing")
    return seqs


def _mean_metrics(rows: List[Dict], weighted: bool = True) -> Dict:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in ("sequence", "scene_name", "valid_pixels")]
    out: Dict = {}
    weights = [float(r.get("valid_pixels", 0)) for r in rows]
    use_w = weighted and all(w > 0 for w in weights)
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r]
        if use_w:
            w = [weights[i] for i, r in enumerate(rows) if k in r]
            num = sum(v * ww for v, ww in zip(vals, w))
            den = sum(w)
            out[k] = float(num / max(den, 1.0))
        else:
            out[k] = float(sum(vals) / max(len(vals), 1))
    out["aggregation"] = "weighted_by_valid_pixels" if use_w else "unweighted_mean"
    return out


# ---------------------------------------------------------------------------
# Sequence discovery helpers
# ---------------------------------------------------------------------------

def _discover_bonn(image_dir: str) -> List[str]:
    """Return full scene names like rgbd_bonn_balloon2."""
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Bonn image_dir not found: {image_dir}")
    names = []
    for name in sorted(os.listdir(image_dir)):
        if not name.startswith("rgbd_bonn_"):
            continue
        if os.path.isdir(os.path.join(image_dir, name, "rgb")):
            names.append(name)
    if not names:
        raise RuntimeError(f"No Bonn scenes found under: {image_dir}")
    return names


def _discover_sintel(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Sintel image_dir not found: {image_dir}")
    seqs = [
        name for name in sorted(os.listdir(image_dir))
        if os.path.isdir(os.path.join(image_dir, name))
        and any(f.endswith(".png") for f in os.listdir(os.path.join(image_dir, name)))
    ]
    if not seqs:
        raise RuntimeError(f"No Sintel sequences found under: {image_dir}")
    return seqs


def _discover_kitti(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"KITTI image_dir not found: {image_dir}")
    seqs = sorted({KITTI_SEQ_RE.search(n).group(0)
                   for n in os.listdir(image_dir)
                   if KITTI_SEQ_RE.search(n)})
    if not seqs:
        raise RuntimeError(f"No KITTI sequences found under: {image_dir}")
    return seqs


def _resolve_sequences_depth(args) -> List[Dict[str, str]]:
    """Returns list of {seq, scene_name} dicts."""
    rows: List[Dict[str, str]] = []

    if args.dataset == "bonn":
        if args.sequences:
            raw = _parse_seq_csv(args.sequences)
            scene_names = [s if s.startswith("rgbd_bonn_") else f"rgbd_bonn_{s}" for s in raw]
        elif args.full_seq:
            scene_names = _discover_bonn(args.image_dir)
        else:
            scene_names = [f"rgbd_bonn_{s}" for s in MONST3R_BONN_SUBSET]
        for sn in scene_names:
            rows.append({"seq": sn.replace("rgbd_bonn_", "", 1), "scene_name": sn})

    elif args.dataset == "sintel":
        if args.sequences:
            seqs = _parse_seq_csv(args.sequences)
        elif args.full_seq:
            seqs = _discover_sintel(args.image_dir)
        else:
            seqs = list(MONST3R_SINTEL_SUBSET)
        rows = [{"seq": s, "scene_name": s} for s in seqs]

    else:  # kitti
        if args.sequences:
            seqs = _parse_seq_csv(args.sequences)
        else:
            seqs = _discover_kitti(args.image_dir)
        rows = [{"seq": s, "scene_name": s} for s in seqs]

    if not rows:
        raise RuntimeError("No sequences resolved")
    return rows


def _resolve_sequences_pose(args) -> List[str]:
    if args.sequences:
        return _parse_seq_csv(args.sequences)
    if args.full_seq:
        return _discover_sintel(args.image_dir)
    return list(MONST3R_SINTEL_SUBSET)


def _resolve_sequences_sf(args) -> List[Dict[str, str]]:
    """single_frame: same logic as depth."""
    return _resolve_sequences_depth(args)


# ---------------------------------------------------------------------------
# Image glob builders
# ---------------------------------------------------------------------------

def _image_glob_depth(args, scene_name: str) -> str:
    if args.dataset == "bonn":
        subdir = "rgb" if args.full_seq else "rgb_110"
        return os.path.join(args.image_dir, scene_name, subdir, "*.png")
    if args.dataset == "kitti":
        return os.path.join(args.image_dir, f"{scene_name}*.png")
    return os.path.join(args.image_dir, scene_name, "*.png")


# ---------------------------------------------------------------------------
# Result JSON name helpers
# ---------------------------------------------------------------------------

def _depth_result_json(dataset: str, scene_name: str, mode: str) -> str:
    if dataset == "bonn":
        base = f"eval_bonn_{scene_name}"
    elif dataset == "kitti":
        base = "eval_kitti"
    else:
        base = f"eval_sintel_{scene_name}"
    suffix = {"scale_and_shift": "_scale_shift", "scale_only": "_scale_only",
              "single_frame": "_single_frame", "median": "_median"}.get(mode, "")
    return base + suffix + ".json"


def _sf_result_json(dataset: str, scene_name: str) -> str:
    if dataset == "bonn":
        return f"eval_bonn_{scene_name}_single_frame.json"
    if dataset == "kitti":
        return "eval_kitti_single_frame.json"
    return f"eval_sintel_{scene_name}_single_frame.json"


# ===========================================================================
# depth subcommand
# ===========================================================================

def _depth_run(args, sequences: List[Dict]) -> int:
    run_py = os.path.join(_PROJECT_DIR, "run.py")
    if not os.path.isfile(run_py):
        raise FileNotFoundError(f"run.py not found: {run_py}")
    if not args.ckpt or not args.raft:
        raise ValueError("--ckpt and --raft are required for stage=run/all")

    os.makedirs(args.output_root, exist_ok=True)
    exit_code = 0

    for item in sequences:
        seq, scene_name = item["seq"], item["scene_name"]
        out_dir = os.path.join(args.output_root, seq)

        if args.skip_existing and os.path.isfile(os.path.join(out_dir, "metrics.json")):
            print(f"[skip] {seq}")
            continue

        image_glob = _image_glob_depth(args, scene_name)
        cmd = [sys.executable, run_py,
               "--images", image_glob, "--ckpt", args.ckpt, "--raft", args.raft,
               "--output", out_dir, "--preprocess", args.preprocess,
               "--niter", str(args.niter), "--loss_version", args.loss_version,
               "--device", args.device]
        if args.max_frames:
            cmd += ["--max_frames", str(args.max_frames)]
        if args.verbose:
            cmd += ["--verbose"]

        rc = _run_cmd(cmd, args.dry_run)
        if rc != 0:
            print(f"[error] run failed: {seq} (exit={rc})")
            exit_code = rc
            if not args.continue_on_error:
                return exit_code
    return exit_code


def _depth_eval(args, sequences: List[Dict]) -> int:
    eval_py = os.path.join(_PROJECT_DIR, "eval.py")
    if not os.path.isfile(eval_py):
        raise FileNotFoundError(f"eval.py not found: {eval_py}")

    rows: List[Dict] = []
    exit_code = 0

    for item in sequences:
        seq, scene_name = item["seq"], item["scene_name"]
        out_dir = os.path.join(args.output_root, seq)
        if not os.path.isdir(out_dir):
            print(f"[skip] {seq} (missing output dir)")
            continue

        if args.dataset == "kitti":
            cmd = [sys.executable, eval_py, "kitti",
                   "--output_dir", out_dir, "--gt_dir", args.gt_dir,
                   "--drive", seq, "--align_scale_mode", args.align_scale_mode,
                   "--max_depth", str(args.max_depth)]
            if not args.use_eigen_crop:
                cmd += ["--no_eigen_crop"]
        else:
            scene_dir = os.path.join(args.gt_dir, scene_name)
            cmd = [sys.executable, eval_py, args.dataset,
                   "--output_dir", out_dir, "--scene_dir", scene_dir,
                   "--align_scale_mode", args.align_scale_mode,
                   "--max_depth", str(args.max_depth),
                   "--min_depth", str(args.min_depth)]
            if args.dataset == "sintel":
                cmd += ["--post_clip_max", str(args.post_clip_max)]

        rc = _run_cmd(cmd, args.dry_run)
        if rc != 0:
            print(f"[error] eval failed: {seq} (exit={rc})")
            exit_code = rc
            if not args.continue_on_error:
                return exit_code
            continue

        if args.dry_run:
            continue

        result_path = os.path.join(out_dir, _depth_result_json(
            args.dataset, scene_name, args.align_scale_mode))
        if os.path.isfile(result_path):
            with open(result_path) as f:
                m = json.load(f)
            m["sequence"] = seq
            m["scene_name"] = scene_name
            rows.append(m)
        else:
            print(f"[warn] result json not found: {result_path}")

    if not args.dry_run and rows:
        summary = {
            "dataset": args.dataset, "align_scale_mode": args.align_scale_mode,
            "max_depth": args.max_depth,
            "full_seq": getattr(args, "full_seq", False),
            "num_sequences": len(rows),
            "per_sequence": rows,
            "mean": _mean_metrics(rows),
        }
        sname = f"{args.dataset}_eval_summary_{args.align_scale_mode}.json"
        spath = os.path.join(args.output_root, sname)
        with open(spath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] summary → {spath}")
    return exit_code


# ===========================================================================
# pose subcommand (Sintel only)
# ===========================================================================

def _pose_run(args, sequences: List[str]) -> int:
    run_py = os.path.join(_PROJECT_DIR, "run.py")
    if not os.path.isfile(run_py):
        raise FileNotFoundError(f"run.py not found: {run_py}")
    if not args.ckpt or not args.raft:
        raise ValueError("--ckpt and --raft are required for stage=run/all")

    os.makedirs(args.output_root, exist_ok=True)
    exit_code = 0

    for seq in sequences:
        out_dir = os.path.join(args.output_root, seq)
        if args.skip_existing and os.path.isfile(os.path.join(out_dir, "metrics.json")):
            print(f"[skip] {seq}")
            continue

        image_glob = os.path.join(args.image_dir, seq, "*.png")
        cmd = [sys.executable, run_py,
               "--images", image_glob, "--ckpt", args.ckpt, "--raft", args.raft,
               "--output", out_dir, "--preprocess", args.preprocess,
               "--niter", str(args.niter), "--loss_version", args.loss_version,
               "--device", args.device]
        if args.max_frames:
            cmd += ["--max_frames", str(args.max_frames)]
        if args.verbose:
            cmd += ["--verbose"]

        rc = _run_cmd(cmd, args.dry_run)
        if rc != 0:
            print(f"[error] run failed: {seq} (exit={rc})")
            exit_code = rc
            if not args.continue_on_error:
                return exit_code
    return exit_code


def _pose_eval(args, sequences: List[str]) -> int:
    eval_py = os.path.join(_PROJECT_DIR, "eval.py")
    if not os.path.isfile(eval_py):
        raise FileNotFoundError(f"eval.py not found: {eval_py}")

    rows: List[Dict] = []
    log_lines: List[str] = []
    exit_code = 0

    for seq in sequences:
        out_dir = os.path.join(args.output_root, seq)
        if not os.path.isdir(out_dir):
            print(f"[skip] {seq} (missing output dir)")
            continue

        gt_cam_dir = os.path.join(args.gt_dir, seq)
        result_json = os.path.join(out_dir, "eval_sintel_pose.json")

        cmd = [sys.executable, eval_py, "sintel_pose",
               "--output_dir", out_dir,
               "--gt_cam_dir", gt_cam_dir,
               "--pose_eval_stride", str(args.pose_eval_stride)]

        rc = _run_cmd(cmd, args.dry_run)
        if rc != 0:
            print(f"[error] pose eval failed: {seq} (exit={rc})")
            log_lines.append(f"Sintel-{seq:<16} | EVAL FAILED")
            exit_code = rc
            if not args.continue_on_error:
                break
            continue

        if args.dry_run:
            continue

        if os.path.isfile(result_json):
            with open(result_json) as f:
                m = json.load(f)
            m["sequence"] = seq
            rows.append(m)
            log_lines.append(
                f"Sintel-{seq:<16} | ATE: {m['ate']:.5f}  "
                f"RPE_t: {m['rpe_trans']:.5f}  RPE_r: {m['rpe_rot']:.5f}")
        else:
            log_lines.append(f"Sintel-{seq:<16} | MISSING RESULT JSON")

    if not args.dry_run:
        for line in log_lines:
            print(line)
        if rows:
            avg_ate   = float(sum(r["ate"]       for r in rows) / len(rows))
            avg_rpe_t = float(sum(r["rpe_trans"] for r in rows) / len(rows))
            avg_rpe_r = float(sum(r["rpe_rot"]   for r in rows) / len(rows))
        else:
            avg_ate = avg_rpe_t = avg_rpe_r = 0.0

        summary = {
            "dataset": "sintel", "protocol": "monst3r_pose_aligned",
            "full_seq": args.full_seq,
            "pose_eval_stride": args.pose_eval_stride,
            "num_sequences": len(rows),
            "per_sequence": rows,
            "mean": {"ate": avg_ate, "rpe_trans": avg_rpe_t, "rpe_rot": avg_rpe_r,
                     "aggregation": "unweighted_mean"},
        }
        spath = os.path.join(args.output_root, "sintel_pose_summary.json")
        with open(spath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] summary → {spath}")
    return exit_code


# ===========================================================================
# single_frame subcommand
# ===========================================================================

def _sf_run(args, sequences: List[Dict]) -> int:
    run_py = os.path.join(_SCRIPT_DIR, "vggt_baseline.py")
    if not os.path.isfile(run_py):
        raise FileNotFoundError(f"vggt_baseline.py not found: {run_py}")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for stage=run/all")

    os.makedirs(args.output_root, exist_ok=True)
    exit_code = 0

    for item in sequences:
        seq, scene_name = item["seq"], item["scene_name"]
        out_dir = os.path.join(args.output_root, seq)

        if args.skip_existing and os.path.isfile(os.path.join(out_dir, "manifest.json")):
            print(f"[skip] {seq}")
            continue

        image_glob = _image_glob_depth(args, scene_name)
        cmd = [sys.executable, run_py,
               "--images", image_glob, "--checkpoint", args.checkpoint,
               "--output", out_dir, "--device", args.device]
        if args.max_frames:
            cmd += ["--max_frames", str(args.max_frames)]
        if args.save_conf:
            cmd += ["--save_conf"]

        rc = _run_cmd(cmd, args.dry_run)
        if rc != 0:
            print(f"[error] single-frame run failed: {seq} (exit={rc})")
            exit_code = rc
            if not args.continue_on_error:
                return exit_code
    return exit_code


def _sf_eval(args, sequences: List[Dict]) -> int:
    eval_py = os.path.join(_PROJECT_DIR, "eval.py")
    if not os.path.isfile(eval_py):
        raise FileNotFoundError(f"eval.py not found: {eval_py}")

    rows: List[Dict] = []
    exit_code = 0

    for item in sequences:
        seq, scene_name = item["seq"], item["scene_name"]
        out_dir = os.path.join(args.output_root, seq)
        if not os.path.isdir(out_dir):
            print(f"[skip] {seq} (missing output dir)")
            continue

        if args.dataset == "kitti":
            cmd = [sys.executable, eval_py, "kitti",
                   "--output_dir", out_dir, "--gt_dir", args.gt_dir,
                   "--drive", seq, "--align_scale_mode", "single_frame",
                   "--max_depth", str(args.max_depth)]
            if not args.use_eigen_crop:
                cmd += ["--no_eigen_crop"]
        else:
            scene_dir = os.path.join(args.gt_dir, scene_name)
            cmd = [sys.executable, eval_py, args.dataset,
                   "--output_dir", out_dir, "--scene_dir", scene_dir,
                   "--align_scale_mode", "single_frame",
                   "--max_depth", str(args.max_depth),
                   "--min_depth", str(args.min_depth)]
            if args.dataset == "sintel":
                cmd += ["--post_clip_max", str(args.post_clip_max)]

        rc = _run_cmd(cmd, args.dry_run)
        if rc != 0:
            print(f"[error] eval failed: {seq} (exit={rc})")
            exit_code = rc
            if not args.continue_on_error:
                return exit_code
            continue

        if args.dry_run:
            continue

        result_path = os.path.join(out_dir, _sf_result_json(args.dataset, scene_name))
        if os.path.isfile(result_path):
            with open(result_path) as f:
                m = json.load(f)
            m["sequence"] = seq
            m["scene_name"] = scene_name
            rows.append(m)
        else:
            print(f"[warn] result json not found: {result_path}")

    if not args.dry_run and rows:
        summary = {
            "dataset": args.dataset, "protocol": "original_vggt_single_frame",
            "align_scale_mode": "single_frame",
            "max_depth": args.max_depth,
            "num_sequences": len(rows),
            "per_sequence": rows,
            "mean": _mean_metrics(rows),
        }
        sname = f"{args.dataset}_vggt_single_frame_summary.json"
        spath = os.path.join(args.output_root, sname)
        with open(spath, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] summary → {spath}")
    return exit_code


# ===========================================================================
# Path resolution helpers
# ===========================================================================

def _abs(path: Optional[str], base: str) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(os.path.join(base, path)) if not os.path.isabs(path) else path


def _resolve_paths_depth(args):
    base = _PROJECT_DIR
    defaults_image = {
        "bonn":   "../data/bonn/rgbd_bonn_dataset",
        "sintel": "../data/sintel/training/final",
        "kitti":  "../data/kitti/depth_selection/val_selection_cropped/image",
    }
    defaults_gt = {
        "bonn":   "../data/bonn/rgbd_bonn_dataset",
        "sintel": "../data/sintel/training/depth",
        "kitti":  "../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth",
    }
    args.image_dir   = _abs(args.image_dir   or defaults_image[args.dataset], base)
    args.gt_dir      = _abs(args.gt_dir      or defaults_gt[args.dataset],    base)
    args.output_root = _abs(args.output_root or f"outputs/{args.dataset}_batch", base)
    if args.ckpt: args.ckpt = _abs(args.ckpt, base)
    if args.raft: args.raft = _abs(args.raft, base)


def _resolve_paths_pose(args):
    base = _PROJECT_DIR
    args.image_dir   = _abs(args.image_dir   or "../data/sintel/training/final",        base)
    args.gt_dir      = _abs(args.gt_dir      or "../data/sintel/training/camdata_left", base)
    args.output_root = _abs(args.output_root or "outputs/sintel_pose_batch",             base)
    if args.ckpt: args.ckpt = _abs(args.ckpt, base)
    if args.raft: args.raft = _abs(args.raft, base)


def _resolve_paths_sf(args):
    base = _PROJECT_DIR
    defaults_image = {
        "bonn":   "../data/bonn/rgbd_bonn_dataset",
        "sintel": "../data/sintel/training/final",
        "kitti":  "../data/kitti/depth_selection/val_selection_cropped/image",
    }
    defaults_gt = {
        "bonn":   "../data/bonn/rgbd_bonn_dataset",
        "sintel": "../data/sintel/training/depth",
        "kitti":  "../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth",
    }
    args.image_dir   = _abs(args.image_dir   or defaults_image[args.dataset], base)
    args.gt_dir      = _abs(args.gt_dir      or defaults_gt[args.dataset],    base)
    args.output_root = _abs(args.output_root or f"outputs/vggt_single_frame_{args.dataset}", base)
    args.checkpoint  = _abs(args.checkpoint  or "../vggt/checkpoints/VGGT-1B.pt", base)


# ===========================================================================
# CLI
# ===========================================================================

def _add_batch_opts(p: argparse.ArgumentParser):
    p.add_argument("--skip_existing",    action="store_true")
    p.add_argument("--continue_on_error",action="store_true")
    p.add_argument("--dry_run",          action="store_true")
    p.add_argument("--sequences",  default=None,
                   help="comma-separated sequence names")
    p.add_argument("--full_seq",   action="store_true",
                   help="use all discovered sequences (default: MonST3R subset)")


def _add_run_opts(p: argparse.ArgumentParser):
    p.add_argument("--ckpt",   default=None, help="VGGT checkpoint (.pt)")
    p.add_argument("--raft",   default=None, help="RAFT weights (.pth)")
    p.add_argument("--preprocess", choices=["letterbox", "center_crop", "long_edge"],
                   default="long_edge")
    p.add_argument("--niter",       type=int,   default=50)
    p.add_argument("--loss_version",choices=["mon", "dyn"], default="mon")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--max_frames",  type=int, default=None)
    p.add_argument("--verbose",     action="store_true")


def _add_depth_eval_opts(p: argparse.ArgumentParser):
    p.add_argument("--align_scale_mode",
                   choices=["scale_only", "scale_and_shift", "single_frame", "median", "none"],
                   default="scale_and_shift")
    p.add_argument("--max_depth",  type=float, default=70.0)
    p.add_argument("--min_depth",  type=float, default=0.0)
    p.add_argument("--post_clip_max", type=float, default=70.0,
                   help="Sintel: post-clip predicted depth before eval")
    p.add_argument("--use_eigen_crop", action="store_true",
                   help="KITTI: enable Eigen crop (default off to match MonST3R)")


def parse_args():
    top = argparse.ArgumentParser(
        description="VGGT-Dyn unified batch runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = top.add_subparsers(dest="subcommand", required=True)

    # ── depth ─────────────────────────────────────────────────────────────────
    dp = sub.add_parser("depth",
                        help="run.py + eval.py depth metrics (Bonn/Sintel/KITTI)")
    dp.add_argument("--dataset",  choices=["bonn", "sintel", "kitti"], required=True)
    dp.add_argument("--stage",    choices=["run", "eval", "all"], default="all")
    dp.add_argument("--image_dir", default=None)
    dp.add_argument("--gt_dir",    default=None)
    dp.add_argument("--output_root", default=None)
    _add_batch_opts(dp)
    _add_run_opts(dp)
    _add_depth_eval_opts(dp)

    # ── pose ──────────────────────────────────────────────────────────────────
    pp = sub.add_parser("pose",
                        help="run.py + eval.py sintel_pose  (ATE/RPE)")
    pp.add_argument("--stage",    choices=["run", "eval", "all"], default="all")
    pp.add_argument("--image_dir", default=None, help="Sintel final/ image root")
    pp.add_argument("--gt_dir",    default=None, help="Sintel camdata_left/ root")
    pp.add_argument("--output_root", default=None)
    _add_batch_opts(pp)
    _add_run_opts(pp)
    pp.set_defaults(preprocess="center_crop", niter=300)  # override defaults for pose
    pp.add_argument("--pose_eval_stride", type=int, default=1)

    # ── single_frame ──────────────────────────────────────────────────────────
    sf = sub.add_parser("single_frame",
                        help="scripts/vggt_baseline.py + eval.py  (original VGGT, no TTO)")
    sf.add_argument("--dataset",  choices=["bonn", "sintel", "kitti"], required=True)
    sf.add_argument("--stage",    choices=["run", "eval", "all"], default="all")
    sf.add_argument("--image_dir",  default=None)
    sf.add_argument("--gt_dir",     default=None)
    sf.add_argument("--output_root",default=None)
    _add_batch_opts(sf)
    sf.add_argument("--checkpoint", default=None, help="VGGT checkpoint (.pt)")
    sf.add_argument("--device",     default="cuda")
    sf.add_argument("--max_frames", type=int, default=None)
    sf.add_argument("--save_conf",  action="store_true")
    _add_depth_eval_opts(sf)

    return top.parse_args()


def main():
    args = parse_args()

    if args.subcommand == "depth":
        _resolve_paths_depth(args)
        sequences = _resolve_sequences_depth(args)
        print(f"[info] depth  dataset={args.dataset}  sequences={len(sequences)}")
        for item in sequences:
            print(f"  - {item['seq']}")
        rc = 0
        if args.stage in ("run", "all"):
            rc = _depth_run(args, sequences)
            if rc != 0 and not args.continue_on_error:
                sys.exit(rc)
        if args.stage in ("eval", "all"):
            rc2 = _depth_eval(args, sequences)
            rc = rc or rc2

    elif args.subcommand == "pose":
        _resolve_paths_pose(args)
        sequences = _resolve_sequences_pose(args)
        print(f"[info] pose  sintel  sequences={len(sequences)}")
        for s in sequences:
            print(f"  - {s}")
        rc = 0
        if args.stage in ("run", "all"):
            rc = _pose_run(args, sequences)
            if rc != 0 and not args.continue_on_error:
                sys.exit(rc)
        if args.stage in ("eval", "all"):
            rc2 = _pose_eval(args, sequences)
            rc = rc or rc2

    elif args.subcommand == "single_frame":
        _resolve_paths_sf(args)
        sequences = _resolve_sequences_sf(args)
        print(f"[info] single_frame  dataset={args.dataset}  sequences={len(sequences)}")
        for item in sequences:
            print(f"  - {item['seq']}")
        rc = 0
        if args.stage in ("run", "all"):
            rc = _sf_run(args, sequences)
            if rc != 0 and not args.continue_on_error:
                sys.exit(rc)
        if args.stage in ("eval", "all"):
            rc2 = _sf_eval(args, sequences)
            rc = rc or rc2

    sys.exit(rc)


if __name__ == "__main__":
    main()
