#!/usr/bin/env python
"""Collect ATE/RPE from outputs/sweep_loss_weights/<scene>/<tag>/checkpoints/iter_*/eval_sintel_pose.json
into a single CSV/markdown table.

Usage: python scripts/collect_sweep_results.py [output_root]
"""
import json
import os
import sys

OUT_ROOT = sys.argv[1] if len(sys.argv) > 1 else "outputs/sweep_loss_weights"
ITERS = ["iter_0000", "iter_0050", "iter_0100", "iter_0300"]


def load(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d["ate"], d["rpe_trans"], d["rpe_rot"]


def main():
    rows = []
    for scene in sorted(os.listdir(OUT_ROOT)):
        scene_dir = os.path.join(OUT_ROOT, scene)
        if not os.path.isdir(scene_dir):
            continue
        for tag in sorted(os.listdir(scene_dir)):
            tag_dir = os.path.join(scene_dir, tag)
            for it in ITERS:
                p = os.path.join(tag_dir, "checkpoints", it, "eval_sintel_pose.json")
                vals = load(p)
                if vals is None:
                    continue
                ate, rpe_t, rpe_r = vals
                rows.append((scene, tag, it, ate, rpe_t, rpe_r))

    print("scene,tag,iter,ATE,RPE_t,RPE_r")
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]},{r[3]:.5f},{r[4]:.5f},{r[5]:.5f}")


if __name__ == "__main__":
    main()
