"""vggt-monst3r/run.py — Path A: VGGT as pairwise measurer + MonST3R TTO

Replace DUSt3R's pairwise predictions with VGGT pairwise predictions, then
run MonST3R's global PointCloudOptimizer (TTO) unchanged.

Key idea (from README):
  - VGGT(frame_i, frame_j) → world_points in frame_i's camera frame
  - world_points[0] → pred1['pts3d']            (points of frame_i in cam_i frame)
  - world_points[1] → pred2['pts3d_in_other_view'] (points of frame_j in cam_i frame)
  - MonST3R TTO reconciles O(N²) pairwise measurements into a global scene

Usage (run from vggt-dyn/):
    python vggt-monst3r/run.py \\
        --images "data/sintel/training/final/alley_1/*.png" \\
        --ckpt   checkpoints/VGGT-1B.pt \\
        --output outputs/vggt_monst3r/alley_1 \\
        [--raft  checkpoints/raft/raft-things.pth] \\
        [--max_frames 6]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ── sys.path setup ──────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))          # vggt-monst3r/
_VGGT_DYN    = os.path.dirname(_HERE)                               # vggt-dyn/
_REPO_ROOT   = os.path.dirname(_VGGT_DYN)                          # 3D-repo/
_MONST3R_DIR = os.path.join(_REPO_ROOT, "monst3r")                 # 3D-repo/monst3r/
_VGGT_REPO   = os.path.join(_VGGT_DYN, "vggt")                    # vggt-dyn/vggt/

for _p in [_VGGT_DYN, _VGGT_REPO, _MONST3R_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# MonST3R imports (from monst3r/)
from dust3r.image_pairs import make_pairs                           # noqa: E402
from dust3r.utils.image import load_images                          # noqa: E402
from dust3r.utils.device import to_numpy                            # noqa: E402
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode      # noqa: E402

# VGGT imports (from vggt-dyn/vggt/)
from vggt.models.vggt import VGGT                                   # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_vggt(ckpt_path: str, device: torch.device) -> VGGT:
    model = VGGT()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    return model.to(device).eval()


# ═══════════════════════════════════════════════════════════════════════════
# VGGT pairwise adapter
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def vggt_pair_forward(vggt_model: VGGT, img_i_01: torch.Tensor,
                      img_j_01: torch.Tensor, device: torch.device):
    """Run VGGT on a single (frame_i, frame_j) pair.

    Parameters
    ----------
    img_i_01, img_j_01 : [3, H, W] float32 in [0, 1]
        frame_i is treated as cam0 (world origin for this pair).

    Returns
    -------
    pred1 : dict  {'pts3d': [1, H, W, 3], 'conf': [1, H, W]}
    pred2 : dict  {'pts3d_in_other_view': [1, H, W, 3], 'conf': [1, H, W]}

    Coordinate system guarantee (from VGGT training normalization):
        world_points[0] = frame_i's 3-D points in frame_i's camera frame
        world_points[1] = frame_j's 3-D points in frame_i's camera frame
    This maps exactly to MonST3R's pred1/pred2 format with no extra transform.
    """
    H_orig, W_orig = img_i_01.shape[-2:]

    # VGGT patch_size = 14 → input H, W must be multiples of 14
    H14 = (H_orig // 14) * 14
    W14 = (W_orig // 14) * 14

    # [1, 2, 3, H, W]
    imgs = torch.stack([img_i_01, img_j_01]).unsqueeze(0).to(device)

    if H14 != H_orig or W14 != W_orig:
        imgs = F.interpolate(
            imgs.flatten(0, 1), size=(H14, W14),
            mode="bilinear", align_corners=False,
        ).unflatten(0, (1, 2))

    device_type = device.type if hasattr(device, "type") else device.split(":")[0]
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        out = vggt_model(imgs)

    # world_points: [1, 2, H14, W14, 3]  — in frame_i camera frame
    wpts  = out["world_points"].squeeze(0).float()       # [2, H14, W14, 3]
    wconf = out["world_points_conf"].squeeze(0).float()  # [2, H14, W14]

    # Resize pts/conf back to MonST3R's H×W if we had to adjust for patch_size
    if H14 != H_orig or W14 != W_orig:
        wpts = F.interpolate(
            wpts.permute(0, 3, 1, 2),               # [2, 3, H14, W14]
            size=(H_orig, W_orig),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1)                        # [2, H_orig, W_orig, 3]

        wconf = F.interpolate(
            wconf.unsqueeze(1),                      # [2, 1, H14, W14]
            size=(H_orig, W_orig),
            mode="bilinear", align_corners=False,
        ).squeeze(1)                                  # [2, H_orig, W_orig]

    return (
        {"pts3d":               wpts[0:1],  "conf": wconf[0:1]},
        {"pts3d_in_other_view": wpts[1:2],  "conf": wconf[1:2]},
    )


# ═══════════════════════════════════════════════════════════════════════════
# Inference output builder
# ═══════════════════════════════════════════════════════════════════════════

def build_inference_output(vggt_model: VGGT, pairs: list, device: torch.device) -> dict:
    """Replace MonST3R's `inference()` with VGGT pairwise predictions.

    Returns a dict with the same format as MonST3R's `inference()`:
        view1 / view2 : {'idx': list[int], 'img': [E, 3, H, W]}
        pred1         : {'pts3d': [E, H, W, 3], 'conf': [E, H, W]}
        pred2         : {'pts3d_in_other_view': [E, H, W, 3], 'conf': [E, H, W]}
    """
    all_pred1_pts, all_pred1_conf = [], []
    all_pred2_pts, all_pred2_conf = [], []
    idx1_list, idx2_list = [], []
    img1_list, img2_list = [], []

    for view1, view2 in tqdm(pairs, desc="VGGT pairwise forward"):
        i = int(view1["idx"])
        j = int(view2["idx"])

        # MonST3R stores images in [-1, 1]; VGGT expects [0, 1]
        img_i_01 = (view1["img"].squeeze(0) + 1.0) / 2.0   # [3, H, W]
        img_j_01 = (view2["img"].squeeze(0) + 1.0) / 2.0

        pred1, pred2 = vggt_pair_forward(vggt_model, img_i_01, img_j_01, device)

        all_pred1_pts.append(pred1["pts3d"])
        all_pred1_conf.append(pred1["conf"])
        all_pred2_pts.append(pred2["pts3d_in_other_view"])
        all_pred2_conf.append(pred2["conf"])
        idx1_list.append(i)
        idx2_list.append(j)
        img1_list.append(view1["img"])   # keep [-1, 1]: MonST3R RAFT reads these
        img2_list.append(view2["img"])

    return {
        "view1": {
            "idx": idx1_list,
            "img": torch.cat(img1_list, dim=0),   # [E, 3, H, W]
        },
        "view2": {
            "idx": idx2_list,
            "img": torch.cat(img2_list, dim=0),
        },
        "pred1": {
            "pts3d": torch.cat(all_pred1_pts,  dim=0),   # [E, H, W, 3]
            "conf":  torch.cat(all_pred1_conf, dim=0),   # [E, H, W]
        },
        "pred2": {
            "pts3d_in_other_view": torch.cat(all_pred2_pts,  dim=0),
            "conf":               torch.cat(all_pred2_conf, dim=0),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Output saving
# ═══════════════════════════════════════════════════════════════════════════

def save_outputs(scene, output_dir: str, image_paths: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)

    # Camera-to-world poses [S, 4, 4]
    c2w = to_numpy(scene.get_im_poses())
    np.save(os.path.join(output_dir, "poses_c2w.npy"), c2w)

    # Cam-from-world extrinsics [S, 3, 4] (OpenCV convention, matches evaluators)
    extrinsics = np.linalg.inv(c2w)[:, :3, :]
    np.save(os.path.join(output_dir, "extrinsics.npy"), extrinsics)

    # Intrinsics [S, 3, 3]
    K = to_numpy(scene.get_intrinsics())
    np.save(os.path.join(output_dir, "intrinsics.npy"), K)

    # Per-frame depth maps
    depths = to_numpy(scene.get_depthmaps())
    for i, d in enumerate(depths):
        np.save(os.path.join(depth_dir, f"frame_{i:04d}.npy"), d)

    # 3-D point cloud [S, H, W, 3]
    pts3d_list = to_numpy(scene.get_pts3d())
    pts3d = np.stack(pts3d_list, axis=0)
    np.save(os.path.join(output_dir, "pts3d.npy"), pts3d)

    # TUM trajectory
    scene.save_tum_poses(os.path.join(output_dir, "pred_traj.txt"))

    meta = {
        "image_paths": [str(p) for p in image_paths],
        "num_frames":  len(image_paths),
        "method":      "vggt-monst3r",
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  extrinsics.npy  {extrinsics.shape}")
    print(f"  intrinsics.npy  {K.shape}")
    print(f"  pts3d.npy       {pts3d.shape}")
    print(f"  depth/          {len(depths)} frames")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="VGGT + MonST3R TTO (Path A)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images",     required=True,
                   help="Glob pattern for input frames, e.g. 'data/sintel/.../alley_1/*.png'")
    p.add_argument("--ckpt",       required=True,
                   help="VGGT checkpoint (.pt)")
    p.add_argument("--output",     default="outputs/vggt_monst3r",
                   help="Output directory")
    p.add_argument("--raft",       default=None,
                   help="RAFT weights (.pth) — enables flow loss in TTO")
    p.add_argument("--niter",      type=int,   default=300,
                   help="TTO iterations")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--max_frames", type=int,   default=None,
                   help="Cap number of frames (first N)")
    p.add_argument("--scene_graph", default="swinstride-5-noncyclic",
                   help="MonST3R scene graph type")
    p.add_argument("--min_conf_thr", type=float, default=2.0,
                   help="Confidence threshold for TTO (VGGT conf starts at 1)")
    p.add_argument("--flow_loss_weight", type=float, default=0.01,
                   help="Flow loss weight (only active when --raft is provided)")
    p.add_argument("--no_shared_focal", action="store_true", default=False,
                   help="Optimize per-frame focal length (default: shared)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── collect image paths ───────────────────────────────────────────────
    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        raise ValueError(f"No images found matching: {args.images}")
    if args.max_frames:
        image_paths = image_paths[:args.max_frames]
    print(f"Frames: {len(image_paths)}")

    # ── load VGGT ─────────────────────────────────────────────────────────
    print("Loading VGGT …")
    vggt_model = load_vggt(args.ckpt, device)

    # ── load images (MonST3R preprocessing, size=518) ─────────────────────
    # load_images returns list of {'img': [1,3,H,W] in [-1,1], 'idx': ..., ...}
    print("Loading images via MonST3R loader (size=518) …")
    imgs = load_images(image_paths, size=518, verbose=True)

    # ── build pairs ───────────────────────────────────────────────────────
    pairs = make_pairs(imgs, scene_graph=args.scene_graph,
                       prefilter=None, symmetrize=True)
    print(f"Scene graph '{args.scene_graph}': {len(pairs)} pairs (symmetrized)")

    # ── VGGT pairwise forward ─────────────────────────────────────────────
    output = build_inference_output(vggt_model, pairs, device)

    # free VGGT before TTO to reduce peak VRAM
    del vggt_model
    torch.cuda.empty_cache()

    # ── optional: patch RAFT loader for flow loss ─────────────────────────
    flow_loss_weight = 0.0
    if args.raft is not None:
        raft_abs = os.path.abspath(args.raft)
        if not os.path.isfile(raft_abs):
            raise FileNotFoundError(f"RAFT weights not found: {raft_abs}")

        # dust3r.cloud_opt.optimizer imported load_RAFT at module load time.
        # Patch the name in that module's namespace so get_flow() picks it up.
        import dust3r.cloud_opt.optimizer as _opt_mod
        # import the function object to wrap
        from third_party.raft import load_RAFT as _real_load_RAFT

        def _patched_load_RAFT(model_path=None):
            return _real_load_RAFT(raft_abs)

        _opt_mod.load_RAFT = _patched_load_RAFT
        flow_loss_weight = args.flow_loss_weight
        print(f"RAFT enabled  path={raft_abs}  flow_loss_weight={flow_loss_weight}")
    else:
        print("RAFT not provided → flow_loss_weight=0 (pure geometric TTO)")

    # ── global alignment (MonST3R TTO) ───────────────────────────────────
    shared_focal = not args.no_shared_focal
    print(f"\nInitialising PointCloudOptimizer  "
          f"shared_focal={shared_focal}  min_conf_thr={args.min_conf_thr}  "
          f"niter={args.niter} …")

    scene = global_aligner(
        output, device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        shared_focal=shared_focal,
        flow_loss_weight=flow_loss_weight,
        flow_loss_start_epoch=0.1,
        flow_loss_thre=25,
        use_self_mask=False,       # avoids needing extra view fields
        temporal_smoothing_weight=0.01,
        translation_weight=1.0,
        num_total_iter=args.niter,
        min_conf_thr=args.min_conf_thr,
        batchify=True,
    )

    print(f"Running global alignment ({args.niter} iters) …")
    scene.compute_global_alignment(
        init="mst", niter=args.niter, schedule="linear", lr=0.01,
    )

    # ── save outputs ─────────────────────────────────────────────────────
    save_outputs(scene, args.output, image_paths)


if __name__ == "__main__":
    main()
