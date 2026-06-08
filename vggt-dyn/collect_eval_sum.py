#!/usr/bin/env python3
"""Collect evaluation summary JSON files into one directory with unified names.

Examples:
  outputs/bonn_batch/bonn_eval_summary_scale_and_shift.json
    -> outputs/eval_sum/bonn_mon_scale_and_shift.json

  outputs/bonn_batch_dyn/bonn_eval_summary_scale_and_shift.json
    -> outputs/eval_sum/bonn_dyn_scale_and_shift.json
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple


EVAL_SUMMARY_RE = re.compile(
    r"^(?P<dataset>bonn|kitti|sintel)_eval_summary_(?P<mode>[a-zA-Z0-9_\-]+)\.json$"
)
SINGLE_FRAME_SUMMARY_RE = re.compile(
    r"^(?P<dataset>bonn|kitti|sintel)_vggt_single_frame_summary\.json$"
)
POSE_SUMMARY_RE = re.compile(r"^(?P<dataset>sintel)_pose_summary\.json$")


def infer_variant(parent_dir_name: str) -> str:
    name = parent_dir_name.lower()
    if "_dyn" in name:
        return "dyn"
    if "single_frame" in name:
        return "vggt"
    return "mon"


def parse_target_name(src_path: Path) -> Optional[str]:
    filename = src_path.name
    variant = infer_variant(src_path.parent.name)

    m = EVAL_SUMMARY_RE.match(filename)
    if m:
        dataset = m.group("dataset")
        mode = m.group("mode")
        return f"{dataset}_{variant}_{mode}.json"

    m = SINGLE_FRAME_SUMMARY_RE.match(filename)
    if m:
        dataset = m.group("dataset")
        return f"{dataset}_{variant}_single_frame.json"

    m = POSE_SUMMARY_RE.match(filename)
    if m:
        dataset = m.group("dataset")
        return f"{dataset}_{variant}_pose.json"

    return None


def uniquify_name(dst_dir: Path, target_name: str) -> Path:
    dst = dst_dir / target_name
    if not dst.exists():
        return dst

    stem = dst.stem
    suffix = dst.suffix
    idx = 2
    while True:
        cand = dst_dir / f"{stem}_{idx}{suffix}"
        if not cand.exists():
            return cand
        idx += 1


def collect(outputs_dir: Path, eval_sum_dir: Path, dry_run: bool) -> Tuple[int, int]:
    scanned = 0
    copied = 0

    for src in sorted(outputs_dir.rglob("*.json")):
        scanned += 1
        if eval_sum_dir in src.parents:
            continue

        target_name = parse_target_name(src)
        if target_name is None:
            continue

        dst = uniquify_name(eval_sum_dir, target_name)
        print(f"[map] {src} -> {dst}")
        copied += 1

        if not dry_run:
            shutil.copy2(src, dst)

    return scanned, copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect output summary JSON files into eval_sum with unified names"
    )
    parser.add_argument(
        "--outputs_dir",
        default="outputs",
        help="root outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--eval_sum_dir",
        default="outputs/eval_sum",
        help="destination directory for collected summaries (default: outputs/eval_sum)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print mapping only, do not copy files",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    eval_sum_dir = Path(args.eval_sum_dir).resolve()

    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"outputs_dir not found: {outputs_dir}")

    if not args.dry_run:
        eval_sum_dir.mkdir(parents=True, exist_ok=True)

    scanned, copied = collect(outputs_dir, eval_sum_dir, args.dry_run)
    print(f"[done] scanned={scanned}, matched={copied}, dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
