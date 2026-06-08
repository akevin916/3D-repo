#!/usr/bin/env python3
"""Batch runner/evaluator for KITTI in VGGT-Dyn.

Supports three stages:
    run   : run.py on all KITTI drive sequences
    eval  : eval.py kitti on all sequence outputs
    all   : run + eval

This script does NOT change model logic; it orchestrates per-sequence commands.
"""

import os
import re
import sys
import json
import shlex
import argparse
import subprocess
from typing import List, Dict

SEQ_RE = re.compile(r"\d{4}_\d{2}_\d{2}_drive_\d{4}_sync")


def _discover_sequences_from_image_dir(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    seqs = set()
    for name in os.listdir(image_dir):
        m = SEQ_RE.search(name)
        if m:
            seqs.add(m.group(0))

    seqs = sorted(seqs)
    if not seqs:
        raise RuntimeError(f"No KITTI drive sequences found under: {image_dir}")
    return seqs


def _parse_sequences_arg(sequences_arg: str, image_dir: str) -> List[str]:
    if sequences_arg:
        seqs = [s.strip() for s in sequences_arg.split(",") if s.strip()]
        if not seqs:
            raise ValueError("--sequences is empty after parsing")
        return seqs
    return _discover_sequences_from_image_dir(image_dir)


def _print_cmd(cmd: List[str]):
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd))


def _run_cmd(cmd: List[str], dry_run: bool):
    _print_cmd(cmd)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _result_json_name(mode: str) -> str:
    if mode == "scale_and_shift":
        return "eval_kitti_scale_shift.json"
    if mode == "scale_only":
        return "eval_kitti_scale_only.json"
    if mode == "single_frame":
        return "eval_kitti_single_frame.json"
    if mode == "median":
        return "eval_kitti_median.json"
    return "eval_kitti.json"


def _mean_metrics(rows: List[Dict]) -> Dict:
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if k not in ("sequence", "valid_pixels")]
    out = {}
    weights = [float(r.get("valid_pixels", 0)) for r in rows]
    use_weighted = all(w > 0 for w in weights)
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r]
        if use_weighted:
            w = [weights[i] for i, r in enumerate(rows) if k in r]
            num = sum(v * ww for v, ww in zip(vals, w))
            den = sum(w)
            out[k] = float(num / max(den, 1.0))
        else:
            out[k] = float(sum(vals) / max(len(vals), 1))
    out["aggregation"] = "weighted_by_valid_pixels" if use_weighted else "unweighted_mean"
    return out


def stage_run(args, sequences: List[str]) -> int:
    run_py = os.path.join(args.project_dir, "run.py")
    if not os.path.isfile(run_py):
        raise FileNotFoundError(f"run.py not found: {run_py}")

    if not args.ckpt or not args.raft:
        raise ValueError("--ckpt and --raft are required for stage=run/all")

    os.makedirs(args.output_root, exist_ok=True)

    exit_code = 0
    for seq in sequences:
        out_dir = os.path.join(args.output_root, seq)
        metrics_path = os.path.join(out_dir, "metrics.json")
        if args.skip_existing and os.path.isfile(metrics_path):
            print(f"[skip] {seq} (existing metrics.json)")
            continue

        image_glob = os.path.join(args.image_dir, f"{seq}*.png")
        cmd = [
            sys.executable,
            run_py,
            "--images", image_glob,
            "--ckpt", args.ckpt,
            "--raft", args.raft,
            "--output", out_dir,
            "--preprocess", args.preprocess,
            "--niter", str(args.niter),
            "--loss_version", args.loss_version,
            "--device", args.device,
        ]
        if args.max_frames is not None:
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


def stage_eval(args, sequences: List[str]) -> int:
    eval_py = os.path.join(args.project_dir, "eval.py")
    if not os.path.isfile(eval_py):
        raise FileNotFoundError(f"eval.py not found: {eval_py}")
    if not args.gt_dir:
        raise ValueError("--gt_dir is required for stage=eval/all")

    rows = []
    exit_code = 0

    for seq in sequences:
        out_dir = os.path.join(args.output_root, seq)
        if not os.path.isdir(out_dir):
            print(f"[skip] {seq} (missing output dir: {out_dir})")
            continue

        cmd = [
            sys.executable,
            eval_py,
            "kitti",
            "--output_dir", out_dir,
            "--gt_dir", args.gt_dir,
            "--drive", seq,
            "--align_scale_mode", args.align_scale_mode,
            "--max_depth", str(args.max_depth),
        ]
        if not args.use_eigen_crop:
            cmd += ["--no_eigen_crop"]

        rc = _run_cmd(cmd, args.dry_run)
        if rc != 0:
            print(f"[error] eval failed: {seq} (exit={rc})")
            exit_code = rc
            if not args.continue_on_error:
                return exit_code
            continue

        if args.dry_run:
            continue

        result_path = os.path.join(out_dir, _result_json_name(args.align_scale_mode))
        if os.path.isfile(result_path):
            with open(result_path, "r") as f:
                metrics = json.load(f)
            metrics["sequence"] = seq
            rows.append(metrics)
        else:
            print(f"[warn] result json not found: {result_path}")

    if not args.dry_run and rows:
        summary = {
            "align_scale_mode": args.align_scale_mode,
            "max_depth": args.max_depth,
            "no_eigen_crop": (not args.use_eigen_crop),
            "num_sequences": len(rows),
            "per_sequence": rows,
            "mean": _mean_metrics(rows),
        }
        summary_name = f"kitti_eval_summary_{args.align_scale_mode}.json"
        summary_path = os.path.join(args.output_root, summary_name)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] summary saved: {summary_path}")

    return exit_code


def parse_args():
    p = argparse.ArgumentParser(description="KITTI batch runner/evaluator for VGGT-Dyn")
    p.add_argument("--stage", choices=["run", "eval", "all"], default="all",
                   help="run: run.py only; eval: eval.py only; all: both")

    p.add_argument("--project_dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="vggt-dyn project directory (default: this script's dir)")
    p.add_argument("--image_dir",
                   default="../data/kitti/depth_selection/val_selection_cropped/image",
                   help="KITTI image directory containing all drive frames")
    p.add_argument("--gt_dir",
                   default="../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth",
                   help="KITTI GT depth directory (for eval)")
    p.add_argument("--output_root", default="outputs/kitti_batch",
                   help="root output directory, one subdir per drive sequence")

    p.add_argument("--sequences", default=None,
                   help="comma-separated drive names, e.g. 2011_09_26_drive_0002_sync,2011_09_26_drive_0005_sync")

    # run.py args
    p.add_argument("--ckpt", default=None, help="VGGT checkpoint (.pt)")
    p.add_argument("--raft", default=None, help="RAFT weights (.pth)")
    p.add_argument("--preprocess", choices=["letterbox", "center_crop", "long_edge"], default="long_edge")
    p.add_argument("--niter", type=int, default=50)
    p.add_argument("--loss_version", choices=["mon", "dyn"], default="mon",
                   help="run.py loss version for stage=run/all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--verbose", action="store_true")

    # eval.py args (default = MonST3R-fair protocol)
    p.add_argument("--align_scale_mode",
                   choices=["scale_only", "scale_and_shift", "single_frame"],
                   default="scale_and_shift",
                   help="evaluation mode (per-sequence: scale_only/scale_and_shift; single-frame: single_frame)")
    p.add_argument("--max_depth", type=float, default=0.0,
                   help="depth cap for eval.py (0 means no cap; default follows MonST3R)")
    p.add_argument("--use_eigen_crop", action="store_true",
                   help="enable Eigen crop (default off to match MonST3R protocol)")

    # batch behaviour
    p.add_argument("--skip_existing", action="store_true",
                   help="skip a sequence if run output already exists")
    p.add_argument("--continue_on_error", action="store_true",
                   help="continue remaining sequences when one fails")
    p.add_argument("--dry_run", action="store_true",
                   help="print commands only, do not execute")

    return p.parse_args()


def main():
    args = parse_args()

    project_dir = os.path.abspath(args.project_dir)
    image_dir = os.path.abspath(os.path.join(project_dir, args.image_dir)) if not os.path.isabs(args.image_dir) else args.image_dir
    gt_dir = os.path.abspath(os.path.join(project_dir, args.gt_dir)) if not os.path.isabs(args.gt_dir) else args.gt_dir
    output_root = os.path.abspath(os.path.join(project_dir, args.output_root)) if not os.path.isabs(args.output_root) else args.output_root

    args.project_dir = project_dir
    args.image_dir = image_dir
    args.gt_dir = gt_dir
    args.output_root = output_root

    sequences = _parse_sequences_arg(args.sequences, args.image_dir)
    print(f"[info] sequences ({len(sequences)}):")
    for s in sequences:
        print(f"  - {s}")

    rc = 0
    if args.stage in ("run", "all"):
        rc = stage_run(args, sequences)
        if rc != 0 and not args.continue_on_error:
            sys.exit(rc)

    if args.stage in ("eval", "all"):
        rc2 = stage_eval(args, sequences)
        if rc == 0:
            rc = rc2

    sys.exit(rc)


if __name__ == "__main__":
    main()
