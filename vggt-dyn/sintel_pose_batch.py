#!/usr/bin/env python3
"""Batch runner for Sintel pose evaluation aligned with MonST3R protocol.

Stages:
  run  : run.py on sequences
  eval : sintel_pose_eval.py on sequence outputs
  all  : run + eval

Defaults align with MonST3R Sintel pose setup:
- sequence subset: 14-sequence Sintel list
- full sequence mode via --full_seq
- pose_eval_stride default 1
- unweighted mean over sequences for final summary
"""

import os
import sys
import json
import shlex
import argparse
import subprocess
from typing import Dict, List

MONST3R_SINTEL_SUBSET = [
    "alley_2",
    "ambush_4",
    "ambush_5",
    "ambush_6",
    "cave_2",
    "cave_4",
    "market_2",
    "market_5",
    "market_6",
    "shaman_3",
    "sleeping_1",
    "sleeping_2",
    "temple_2",
    "temple_3",
]


def _print_cmd(cmd: List[str]):
    print("[cmd] " + " ".join(shlex.quote(x) for x in cmd))


def _run_cmd(cmd: List[str], dry_run: bool):
    _print_cmd(cmd)
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def _parse_seq_csv(sequences_arg: str) -> List[str]:
    seqs = [s.strip() for s in (sequences_arg or "").split(",") if s.strip()]
    if not seqs:
        raise ValueError("--sequences is empty after parsing")
    return seqs


def _discover_sintel_full_sequences(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Sintel image_dir not found: {image_dir}")

    seqs = []
    for name in sorted(os.listdir(image_dir)):
        seq_dir = os.path.join(image_dir, name)
        if not os.path.isdir(seq_dir):
            continue
        has_png = any(f.endswith(".png") for f in os.listdir(seq_dir))
        if has_png:
            seqs.append(name)

    if not seqs:
        raise RuntimeError(f"No Sintel sequences found under: {image_dir}")
    return seqs


def _resolve_sequences(args) -> List[str]:
    if args.sequences:
        return _parse_seq_csv(args.sequences)
    if args.full_seq:
        return _discover_sintel_full_sequences(args.image_dir)
    return list(MONST3R_SINTEL_SUBSET)


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

        image_glob = os.path.join(args.image_dir, seq, "*.png")
        cmd = [
            sys.executable,
            run_py,
            "--images",
            image_glob,
            "--ckpt",
            args.ckpt,
            "--raft",
            args.raft,
            "--output",
            out_dir,
            "--preprocess",
            args.preprocess,
            "--niter",
            str(args.niter),
            "--loss_version",
            args.loss_version,
            "--device",
            args.device,
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
    eval_py = os.path.join(args.project_dir, "sintel_pose_eval.py")
    if not os.path.isfile(eval_py):
        raise FileNotFoundError(f"sintel_pose_eval.py not found: {eval_py}")

    rows = []
    exit_code = 0
    log_lines = []

    for seq in sequences:
        out_dir = os.path.join(args.output_root, seq)
        if not os.path.isdir(out_dir):
            print(f"[skip] {seq} (missing output dir: {out_dir})")
            continue

        gt_cam_dir = os.path.join(args.gt_dir, seq)
        result_json = os.path.join(out_dir, "eval_sintel_pose.json")

        cmd = [
            sys.executable,
            eval_py,
            "--output_dir",
            out_dir,
            "--gt_cam_dir",
            gt_cam_dir,
            "--pose_eval_stride",
            str(args.pose_eval_stride),
        ]

        rc = _run_cmd(cmd, args.dry_run)
        if rc != 0:
            print(f"[error] eval failed: {seq} (exit={rc})")
            log_lines.append(f"Sintel-{seq:<16} | EVAL FAILED")
            exit_code = rc
            if not args.continue_on_error:
                break
            continue

        if args.dry_run:
            continue

        if os.path.isfile(result_json):
            with open(result_json, "r") as f:
                metrics = json.load(f)
            metrics["sequence"] = seq
            rows.append(metrics)
            log_lines.append(
                f"Sintel-{seq:<16} | ATE: {metrics['ate']:.5f}, "
                f"RPE trans: {metrics['rpe_trans']:.5f}, RPE rot: {metrics['rpe_rot']:.5f}"
            )
        else:
            log_lines.append(f"Sintel-{seq:<16} | MISSING RESULT JSON")

    if not args.dry_run:
        error_log = os.path.join(args.output_root, "_error_log.txt")

        if rows:
            avg_ate = float(sum(r["ate"] for r in rows) / len(rows))
            avg_rpe_t = float(sum(r["rpe_trans"] for r in rows) / len(rows))
            avg_rpe_r = float(sum(r["rpe_rot"] for r in rows) / len(rows))
        else:
            avg_ate, avg_rpe_t, avg_rpe_r = 0.0, 0.0, 0.0

        summary = {
            "dataset": "sintel",
            "protocol": "monst3r_pose_aligned",
            "full_seq": args.full_seq,
            "sequence_scope": "full_seq" if args.full_seq else "monst3r_subset",
            "pose_eval_stride": args.pose_eval_stride,
            "num_sequences": len(rows),
            "per_sequence": rows,
            "mean": {
                "ate": avg_ate,
                "rpe_trans": avg_rpe_t,
                "rpe_rot": avg_rpe_r,
                "aggregation": "unweighted_mean_over_sequences",
            },
        }

        summary_path = os.path.join(args.output_root, "sintel_pose_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        with open(error_log, "w") as f:
            for line in log_lines:
                f.write(line + "\n")
            f.write(
                f"Average ATE: {avg_ate:.5f}, Average RPE trans: {avg_rpe_t:.5f}, "
                f"Average RPE rot: {avg_rpe_r:.5f}\n"
            )

        print(f"[done] summary saved: {summary_path}")
        print(f"[done] error log saved: {error_log}")

    return exit_code


def parse_args():
    p = argparse.ArgumentParser(description="Sintel pose batch runner/evaluator (MonST3R-aligned)")
    p.add_argument("--stage", choices=["run", "eval", "all"], default="all",
                   help="run: run.py only; eval: pose eval only; all: both")

    p.add_argument("--project_dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="vggt-dyn project directory (default: this script's dir)")
    p.add_argument("--image_dir", default="../data/sintel/training/final",
                   help="Sintel image root")
    p.add_argument("--gt_dir", default="../data/sintel/training/camdata_left",
                   help="Sintel GT camera root")
    p.add_argument("--output_root", default="outputs/sintel_pose_batch",
                   help="root output directory")

    p.add_argument("--sequences", default=None,
                   help="comma-separated sequence names")
    p.add_argument("--full_seq", action="store_true",
                   help="evaluate all sequences discovered from image_dir (default: MonST3R subset)")

    # run.py args
    p.add_argument("--ckpt", default=None, help="VGGT checkpoint (.pt)")
    p.add_argument("--raft", default=None, help="RAFT weights (.pth)")
    p.add_argument("--preprocess", choices=["letterbox", "center_crop", "long_edge"], default="center_crop",
                   help="MonST3R pose default uses crop; center_crop is the closest option")
    p.add_argument("--niter", type=int, default=300, help="run.py optimization iterations")
    p.add_argument("--loss_version", choices=["mon", "dyn"], default="mon")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--verbose", action="store_true")

    # pose eval args
    p.add_argument("--pose_eval_stride", type=int, default=1,
                   help="frame stride for pose metrics (MonST3R default: 1)")

    # batch behavior
    p.add_argument("--skip_existing", action="store_true",
                   help="skip a sequence if run output already exists")
    p.add_argument("--continue_on_error", action="store_true",
                   help="continue remaining sequences when one fails")
    p.add_argument("--dry_run", action="store_true",
                   help="print commands only, do not execute")

    return p.parse_args()


def _resolve_paths(args):
    project_dir = os.path.abspath(args.project_dir)

    args.project_dir = project_dir
    args.image_dir = os.path.abspath(os.path.join(project_dir, args.image_dir)) if not os.path.isabs(args.image_dir) else args.image_dir
    args.gt_dir = os.path.abspath(os.path.join(project_dir, args.gt_dir)) if not os.path.isabs(args.gt_dir) else args.gt_dir
    args.output_root = os.path.abspath(os.path.join(project_dir, args.output_root)) if not os.path.isabs(args.output_root) else args.output_root
    if args.ckpt and (not os.path.isabs(args.ckpt)):
        args.ckpt = os.path.abspath(os.path.join(project_dir, args.ckpt))
    if args.raft and (not os.path.isabs(args.raft)):
        args.raft = os.path.abspath(os.path.join(project_dir, args.raft))


def main():
    args = parse_args()
    _resolve_paths(args)

    sequences = _resolve_sequences(args)
    print(f"[info] Sintel sequences ({len(sequences)}), full_seq={args.full_seq}")
    for seq in sequences:
        print(f"  - {seq}")

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
