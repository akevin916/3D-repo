"""run.py — CLI entry point for VGGT-Dyn test-time optimization.

Usage:
    python run.py \\
        --images   path/to/frames/*.png \\
        --ckpt     vggt/checkpoints/VGGT-1B.pt \\
        --raft     monst3r/third_party/RAFT/models/raft-things.pth \\
        --output   outputs/my_scene \\
        --niter    50 \\
        --device   cuda

Output (all saved under --output):
    depth/           per-frame refined depth [H, W] as .npy
    extrinsics.npy   [S, 3, 4] cam-from-world
    intrinsics.npy   [S, 3, 3] camera intrinsics
    pts3d.npy        [S, H, W, 3] refined world-frame 3D points
    dynamic_mask/    per-frame dynamic mask [H, W] as .npy (bool)
    metrics.json     final loss values

Core pipeline logic lives in vggt_dyn/pipeline.py and can be imported
programmatically without going through this CLI.
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_VGGT_REPO = os.path.join(_SCRIPT_DIR, "vggt")
if _VGGT_REPO not in sys.path:
    sys.path.insert(0, _VGGT_REPO)

from vggt_dyn.pipeline import run_pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VGGT-Dyn: test-time optimization for dynamic scenes")

    # I/O
    p.add_argument("--images",      required=True,  help="glob pattern for input frames (e.g. 'frames/*.png')")
    p.add_argument("--ckpt",        required=True,  help="VGGT checkpoint (.pt)")
    p.add_argument("--raft",        required=True,  help="RAFT model weights (.pth)")
    p.add_argument("--output",      default="outputs/vggt_dyn", help="output directory")
    p.add_argument("--max_frames",  type=int, default=None,
                   help="cap number of frames (first N); useful to avoid OOM on long sequences")
    p.add_argument("--preprocess", default="letterbox",
                   choices=["letterbox", "center_crop", "long_edge"],
                   help="image preprocessing: "
                        "'letterbox' (pad to square, default), "
                        "'center_crop' (crop center square), "
                        "'long_edge' (resize long edge to 518, keep aspect ratio — closest to MonST3R)")

    # Optimization
    p.add_argument("--niter",        type=int,   default=50,    help="number of optimization iterations")
    p.add_argument("--lr",           type=float, default=1e-3,  help="peak learning rate (cosine schedule)")
    p.add_argument("--lr_min",       type=float, default=1e-5,  help="minimum learning rate")
    p.add_argument("--loss_version", default="mon", choices=["mon", "dyn"],
                   help="mon: anchor+flow+depth_reg (original)  "
                        "dyn: static_rigid+dyn_pointmap+track_smooth")
    # mon weights
    p.add_argument("--anchor_weight",     type=float, default=1.0,  help="[mon] anchor loss weight")
    p.add_argument("--flow_weight",       type=float, default=1.0,  help="[mon] flow loss weight")
    p.add_argument("--depth_reg_weight",  type=float, default=0.1,  help="[mon] depth regularization weight")
    p.add_argument("--smooth_weight",        type=float, default=0.1,  help="[mon] pose smooth loss weight")
    p.add_argument("--translation_weight",  type=float, default=0.1,
                   help="[mon] rotation-vs-translation balance inside pose smooth loss")
    p.add_argument("--flow_loss_start_frac", type=float, default=0.15,
                   help="[mon] skip flow loss for first this fraction of iters (warmup)")
    p.add_argument("--flow_loss_thre",      type=float, default=50.0,
                   help="[mon] disable flow loss for this iter if loss > threshold (0=off)")
    p.add_argument("--pxl_thre",            type=float, default=50.0,
                   help="[mon] per-pixel flow loss outlier cutoff in pixels (0=off)")
    # dyn weights
    p.add_argument("--dyn_pointmap_weight", type=float, default=1.0,
                   help="[dyn] λ — dynamic pointmap regression weight")
    p.add_argument("--track_smooth_weight", type=float, default=0.1,
                   help="[dyn] γ — 3D track smoothness weight")

    # Dynamic mask
    p.add_argument("--mask_refresh",   type=int,   default=10,   help="refresh dynamic mask every N iters")
    p.add_argument("--mask_threshold", type=float, default=0.35, help="normalized flow-residual threshold")
    p.add_argument("--gt_mask_dir",    default=None,
                   help="directory containing GT dynamic-mask PNGs (frame_XXXX.png, binary). "
                        "If set, masks are loaded from disk at init and NOT recomputed during TTO. "
                        "Expected file names must match the basenames of --images.")

    # Intermediate evaluation
    p.add_argument("--eval_every", type=int, default=0,
                   help="if >0, save extrinsics/depth checkpoints every N iters "
                        "to <output>/checkpoints/iter_XXXX/ for intermediate eval")

    p.add_argument("--freeze_pose", action="store_true",
                   help="freeze delta_rotvec/delta_t (pose not optimized; depth-only TTO)")
    p.add_argument("--scale_flow_loss", action="store_true",
                   help="Exp C: per-pair stop-gradient scale normalization on flow loss "
                        "(removes ego_flow magnitude bias from TTO gradients)")

    # Misc
    p.add_argument("--device",  default="cuda")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
