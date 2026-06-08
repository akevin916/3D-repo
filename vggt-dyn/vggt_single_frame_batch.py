#!/usr/bin/env python3
"""Batch runner for original VGGT single-frame depth evaluation.

Pipeline per sequence:
  1) run_vggt_single_frame.py (pure feed-forward VGGT per frame)
  2) eval.py <dataset> with --align_scale_mode single_frame

Supported datasets:
  - bonn
  - sintel

Default sequence scope follows MonST3R subset protocol.
"""

import os
import re
import sys
import json
import shlex
import argparse
import subprocess
from typing import Dict, List

MONST3R_BONN_SUBSET = [
    "balloon2",
    "crowd2",
    "crowd3",
    "person_tracking2",
    "synchronous",
]

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

KITTI_SEQ_RE = re.compile(r"\d{4}_\d{2}_\d{2}_drive_\d{4}_sync")


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


def _discover_bonn_full_sequences(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Bonn image_dir not found: {image_dir}")

    scene_names = []
    for name in sorted(os.listdir(image_dir)):
        scene_dir = os.path.join(image_dir, name)
        if not os.path.isdir(scene_dir):
            continue
        if not name.startswith("rgbd_bonn_"):
            continue
        rgb_dir = os.path.join(scene_dir, "rgb")
        if os.path.isdir(rgb_dir):
            scene_names.append(name)

    if not scene_names:
        raise RuntimeError(f"No Bonn scenes found under: {image_dir}")
    return scene_names


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


def _discover_kitti_sequences(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"KITTI image_dir not found: {image_dir}")

    seqs = set()
    for name in os.listdir(image_dir):
        m = KITTI_SEQ_RE.search(name)
        if m:
            seqs.add(m.group(0))

    seqs = sorted(seqs)
    if not seqs:
        raise RuntimeError(f"No KITTI sequences found under: {image_dir}")
    return seqs


def _resolve_sequences(args) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    if args.dataset == "bonn":
        if args.sequences:
            input_seqs = _parse_seq_csv(args.sequences)
            if args.full_seq:
                scene_names = [s if s.startswith("rgbd_bonn_") else f"rgbd_bonn_{s}" for s in input_seqs]
            else:
                scene_names = [f"rgbd_bonn_{s}" for s in input_seqs]
        elif args.full_seq:
            scene_names = _discover_bonn_full_sequences(args.image_dir)
        else:
            scene_names = [f"rgbd_bonn_{s}" for s in MONST3R_BONN_SUBSET]

        for scene_name in scene_names:
            if not scene_name.startswith("rgbd_bonn_"):
                raise ValueError(f"Invalid Bonn scene name: {scene_name}")
            seq = scene_name.replace("rgbd_bonn_", "", 1)
            rows.append({"seq": seq, "scene_name": scene_name})

    elif args.dataset == "sintel":
        if args.sequences:
            seqs = _parse_seq_csv(args.sequences)
        elif args.full_seq:
            seqs = _discover_sintel_full_sequences(args.image_dir)
        else:
            seqs = list(MONST3R_SINTEL_SUBSET)

        for seq in seqs:
            rows.append({"seq": seq, "scene_name": seq})

    else:
        if args.sequences:
            seqs = _parse_seq_csv(args.sequences)
        else:
            seqs = _discover_kitti_sequences(args.image_dir)

        for seq in seqs:
            rows.append({"seq": seq, "scene_name": seq})

    if not rows:
        raise RuntimeError("No sequences resolved")
    return rows


def _result_json_name(dataset: str, scene_name: str) -> str:
    if dataset == "bonn":
        return f"eval_bonn_{scene_name}_single_frame.json"
    if dataset == "kitti":
        return "eval_kitti_single_frame.json"
    return f"eval_sintel_{scene_name}_single_frame.json"


def _mean_metrics(rows: List[Dict]) -> Dict:
    if not rows:
        return {}

    keys = [k for k in rows[0].keys() if k not in ("sequence", "scene_name", "valid_pixels")]
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


def _run_image_glob(args, scene_name: str) -> str:
    if args.dataset == "bonn":
        rgb_subdir = "rgb" if args.full_seq else "rgb_110"
        return os.path.join(args.image_dir, scene_name, rgb_subdir, "*.png")
    if args.dataset == "kitti":
        return os.path.join(args.image_dir, f"{scene_name}*.png")
    return os.path.join(args.image_dir, scene_name, "*.png")


def _eval_scene_dir(args, scene_name: str) -> str:
    return os.path.join(args.gt_dir, scene_name)


def stage_run(args, sequences: List[Dict[str, str]]) -> int:
    run_py = os.path.join(args.project_dir, "run_vggt_single_frame.py")
    if not os.path.isfile(run_py):
        raise FileNotFoundError(f"run_vggt_single_frame.py not found: {run_py}")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for stage=run/all")

    os.makedirs(args.output_root, exist_ok=True)

    exit_code = 0
    for item in sequences:
        seq = item["seq"]
        scene_name = item["scene_name"]

        out_dir = os.path.join(args.output_root, seq)
        marker = os.path.join(out_dir, "manifest.json")
        if args.skip_existing and os.path.isfile(marker):
            print(f"[skip] {seq} (existing manifest.json)")
            continue

        image_glob = _run_image_glob(args, scene_name)
        cmd = [
            sys.executable,
            run_py,
            "--images",
            image_glob,
            "--checkpoint",
            args.checkpoint,
            "--output",
            out_dir,
            "--device",
            args.device,
        ]
        if args.max_frames is not None:
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


def stage_eval(args, sequences: List[Dict[str, str]]) -> int:
    eval_py = os.path.join(args.project_dir, "eval.py")
    if not os.path.isfile(eval_py):
        raise FileNotFoundError(f"eval.py not found: {eval_py}")

    rows = []
    exit_code = 0

    for item in sequences:
        seq = item["seq"]
        scene_name = item["scene_name"]

        out_dir = os.path.join(args.output_root, seq)
        if not os.path.isdir(out_dir):
            print(f"[skip] {seq} (missing output dir: {out_dir})")
            continue

        if args.dataset == "kitti":
            cmd = [
                sys.executable,
                eval_py,
                "kitti",
                "--output_dir",
                out_dir,
                "--gt_dir",
                args.gt_dir,
                "--drive",
                seq,
                "--align_scale_mode",
                "single_frame",
                "--max_depth",
                str(args.max_depth),
            ]
            if not args.use_eigen_crop:
                cmd += ["--no_eigen_crop"]
        else:
            scene_dir = _eval_scene_dir(args, scene_name)
            cmd = [
                sys.executable,
                eval_py,
                args.dataset,
                "--output_dir",
                out_dir,
                "--scene_dir",
                scene_dir,
                "--align_scale_mode",
                "single_frame",
                "--max_depth",
                str(args.max_depth),
                "--min_depth",
                str(args.min_depth),
            ]

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

        result_path = os.path.join(out_dir, _result_json_name(args.dataset, scene_name))
        if os.path.isfile(result_path):
            with open(result_path, "r") as f:
                metrics = json.load(f)
            metrics["sequence"] = seq
            metrics["scene_name"] = scene_name
            rows.append(metrics)
        else:
            print(f"[warn] result json not found: {result_path}")

    if not args.dry_run and rows:
        if args.dataset == "kitti":
            sequence_scope = "all_discovered_or_specified"
        else:
            sequence_scope = "full_seq" if args.full_seq else "monst3r_subset"

        summary = {
            "dataset": args.dataset,
            "protocol": "original_vggt_single_frame",
            "align_scale_mode": "single_frame",
            "max_depth": args.max_depth,
            "min_depth": args.min_depth if args.dataset != "kitti" else None,
            "post_clip_max": args.post_clip_max if args.dataset == "sintel" else None,
            "use_eigen_crop": args.use_eigen_crop if args.dataset == "kitti" else None,
            "full_seq": args.full_seq if args.dataset != "kitti" else None,
            "sequence_scope": sequence_scope,
            "num_sequences": len(rows),
            "per_sequence": rows,
            "mean": _mean_metrics(rows),
        }
        summary_name = f"{args.dataset}_vggt_single_frame_summary.json"
        summary_path = os.path.join(args.output_root, summary_name)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] summary saved: {summary_path}")

    return exit_code


def parse_args():
    p = argparse.ArgumentParser(description="Batch original VGGT single-frame depth evaluation")
    p.add_argument("--dataset", choices=["bonn", "sintel", "kitti"], required=True)
    p.add_argument("--stage", choices=["run", "eval", "all"], default="all",
                   help="run: VGGT inference only; eval: eval.py only; all: both")

    p.add_argument("--project_dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="vggt-dyn project directory (default: this script's dir)")

    p.add_argument("--image_dir", default=None,
                   help="input image root (Bonn default: ../data/bonn/rgbd_bonn_dataset; "
                        "Sintel default: ../data/sintel/training/final; "
                        "KITTI default: ../data/kitti/depth_selection/val_selection_cropped/image)")
    p.add_argument("--gt_dir", default=None,
                   help="GT root (Bonn default: ../data/bonn/rgbd_bonn_dataset; "
                        "Sintel default: ../data/sintel/training/depth; "
                        "KITTI default: ../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth)")
    p.add_argument("--output_root", default=None,
                   help="root output directory (default: outputs/vggt_single_frame_{dataset})")

    p.add_argument("--sequences", default=None,
                   help="comma-separated sequence names; for Bonn can be short name (balloon2) or full scene name (rgbd_bonn_balloon2)")
    p.add_argument("--full_seq", action="store_true",
                   help="evaluate all sequences discovered from image_dir (default: MonST3R subset for Bonn/Sintel; all discovered for KITTI)")

    # run_vggt_single_frame.py args
    p.add_argument("--checkpoint", default="../vggt/checkpoints/VGGT-1B.pt", help="VGGT checkpoint (.pt)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--save_conf", action="store_true")

    # eval.py args (single_frame fixed)
    p.add_argument("--max_depth", type=float, default=70.0,
                   help="depth cap for evaluation (MonST3R notebook uses 70 for Bonn/Sintel; KITTI often uses 0 for no cap)")
    p.add_argument("--min_depth", type=float, default=0.0,
                   help="minimum valid depth (MonST3R-style uses gt>0)")
    p.add_argument("--post_clip_max", type=float, default=70.0,
                   help="Sintel only: post-clip predicted depth before eval (MonST3R notebook: 70)")
    p.add_argument("--use_eigen_crop", action="store_true",
                   help="KITTI only: enable Eigen crop (default off to match MonST3R protocol)")

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

    if args.image_dir is None:
        if args.dataset == "bonn":
            image_dir = "../data/bonn/rgbd_bonn_dataset"
        elif args.dataset == "sintel":
            image_dir = "../data/sintel/training/final"
        else:
            image_dir = "../data/kitti/depth_selection/val_selection_cropped/image"
    else:
        image_dir = args.image_dir

    if args.gt_dir is None:
        if args.dataset == "bonn":
            gt_dir = "../data/bonn/rgbd_bonn_dataset"
        elif args.dataset == "sintel":
            gt_dir = "../data/sintel/training/depth"
        else:
            gt_dir = "../data/kitti/depth_selection/val_selection_cropped/groundtruth_depth"
    else:
        gt_dir = args.gt_dir

    if args.output_root is None:
        output_root = f"outputs/vggt_single_frame_{args.dataset}"
    else:
        output_root = args.output_root

    args.project_dir = project_dir
    args.image_dir = os.path.abspath(os.path.join(project_dir, image_dir)) if not os.path.isabs(image_dir) else image_dir
    args.gt_dir = os.path.abspath(os.path.join(project_dir, gt_dir)) if not os.path.isabs(gt_dir) else gt_dir
    args.output_root = os.path.abspath(os.path.join(project_dir, output_root)) if not os.path.isabs(output_root) else output_root
    args.checkpoint = os.path.abspath(os.path.join(project_dir, args.checkpoint)) if not os.path.isabs(args.checkpoint) else args.checkpoint


def main():
    args = parse_args()
    _resolve_paths(args)

    sequences = _resolve_sequences(args)
    print(f"[info] dataset={args.dataset} full_seq={args.full_seq} sequences={len(sequences)}")
    for item in sequences:
        print(f"  - {item['seq']} ({item['scene_name']})")

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
