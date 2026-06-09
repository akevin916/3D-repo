"""eval.py — VGGT-Dyn evaluation dispatcher.

Dispatches to dataset-specific evaluators in evaluators/:

    scared   Surgical scene depth (SCARED, mm-level Chamfer)
    dtu      Multi-view 3D reconstruction (DTU, Chamfer + Sim(3))
    kitti    Monocular depth estimation (KITTI Eigen split, AbsRel/RMSE/δ)

Usage (SCARED):
    python eval.py scared \\
        --output_dir        outputs/my_scene \\
        --gt_cloud          data/SCARED/dataset_8/keyframe_0/point_cloud.obj \\
        --align_scale_mode  median

Usage (DTU — Sim(3) alignment via GT camera centres):
    python eval.py dtu \\
        --output_dir        outputs/my_scene \\
        --stl_dir           data/DTU/Points/stl \\
        --scan_id           24 \\
        --calib_dir         "data/DTU/SampleSet/MVS Data/Calibration/cal18" \\
        --images            "data/DTU/Rectified/scan24/rect_*_3_r5000.png" \\
        --align_scale_mode  sim3

Usage (KITTI):
    python eval.py kitti \\
        --output_dir        outputs/my_kitti_run \\
        --gt_dir            data/KITTI/depth_gt \\
        --align_scale_mode  single_frame

Usage (Sintel):
    python eval.py sintel \\
        --output_dir        outputs/my_sintel_run \\
        --scene_dir         data/sintel/depth/alley_2 \\
        --align_scale_mode  scale_and_shift
"""

import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VGGT-Dyn evaluation")
    sub = p.add_subparsers(dest="mode", required=True)

    # ── 所有子命令共用的 parent parser ────────────────────────────────────────
    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument(
        "--output_dir", required=True,
        help="run.py output directory",
    )
    common_parent.add_argument(
        "--align_scale_mode",
        choices=["none", "single_frame", "median", "scale_only", "scale_and_shift", "sim3"],
        default="none",
        help=(
            "Coordinate alignment before metric computation: "
            "none=no alignment, "
            "single_frame=per-frame median scale + valid-pixel weighted aggregation (MonST3R single-frame), "
            "median=per-frame median scale, "
            "scale_only=per-sequence scale only (no shift), "
            "scale_and_shift=per-sequence scale+shift, "
            "sim3=Umeyama Sim(3) via GT camera centres"
        ),
    )

    # ── SCARED ───────────────────────────────────────────────────────────────
    sp = sub.add_parser("scared", parents=[common_parent],
                        help="SCARED: surgical scene depth (Chamfer, mm)")
    sp.add_argument("--gt_cloud", required=True, help="path to point_cloud.obj")
    sp.add_argument("--gt_calib", default=None,  help="endoscope_calibration.yaml (optional)")

    # ── DTU ──────────────────────────────────────────────────────────────────
    dp = sub.add_parser("dtu", parents=[common_parent],
                        help="DTU: multi-view reconstruction (Chamfer + Sim(3))")
    dp.add_argument("--stl_dir",     required=True, help="DTU Points/stl directory")
    dp.add_argument("--scan_id",     type=int, required=True, help="DTU scan ID (e.g. 24)")
    dp.add_argument("--calib_dir",   default=None,
                    help="dir with pos_XXX.txt (required when --align_scale_mode sim3)")
    dp.add_argument("--images",      default=None,
                    help="glob for input images (required when --align_scale_mode sim3)")
    dp.add_argument("--dataset_dir", default=None,
                    help="DTU SampleSet/MVS Data directory")

    # ── BONN ─────────────────────────────────────────────────────────────────
    bp = sub.add_parser("bonn", parents=[common_parent],
                        help="Bonn RGB-D: dynamic indoor depth (AbsRel/RMSE/δ)")
    bp.add_argument("--scene_dir", required=True,
                    help="Bonn scene directory, e.g. data/bonn/rgbd_bonn_dataset/rgbd_bonn_balloon")
    bp.add_argument("--max_depth", type=float, default=10.0,
                    help="depth cap in metres (default: 10)")
    bp.add_argument("--min_depth", type=float, default=0.1,
                    help="minimum valid depth in metres (default: 0.1)")

    # ── SINTEL ───────────────────────────────────────────────────────────────
    sp_sintel = sub.add_parser("sintel", parents=[common_parent],
                               help="MPI-Sintel: monocular depth (AbsRel/RMSE/δ)")
    sp_sintel.add_argument("--scene_dir", required=True,
                           help="Sintel scene depth directory, e.g. data/sintel/depth/alley_2")
    sp_sintel.add_argument("--max_depth", type=float, default=70.0,
                           help="depth cap in metres (default: 70, use 0 for no cap)")
    sp_sintel.add_argument("--min_depth", type=float, default=0.0,
                           help="minimum valid depth in metres (default: 0)")
    sp_sintel.add_argument("--post_clip_max", type=float, default=70.0,
                           help="clip predicted depth before evaluation (default: 70, use 0 to disable)")

    # ── KITTI ────────────────────────────────────────────────────────────────
    kp = sub.add_parser("kitti", parents=[common_parent],
                        help="KITTI: monocular depth (AbsRel/RMSE/δ, Eigen split)")
    kp.add_argument("--gt_dir",        required=True,
                    help="GT depth dir: .npy (float32, metres) or .png (uint16/256=metres)")
    kp.add_argument("--max_depth",     type=float, default=80.0,
                    help="depth cap in metres (default: 80, use 0 for no cap)")
    kp.add_argument("--no_eigen_crop", action="store_true",
                    help="disable Eigen crop (evaluate full image)")
    kp.add_argument("--drive",         default=None,
                    help="KITTI drive name to filter GT files, e.g. "
                         "'2011_09_26_drive_0002_sync'")

    # ── SINTEL POSE ──────────────────────────────────────────────────────────
    spp = sub.add_parser("sintel_pose", parents=[common_parent],
                         help="Sintel pose evaluation (ATE/RPE)")
    spp.add_argument("--gt_cam_dir",       required=True,
                     help="Sintel camdata_left/<seq> directory")
    spp.add_argument("--gt_traj_dir",      default=None,
                     help="deprecated alias of --gt_cam_dir")
    spp.add_argument("--pose_eval_stride", type=int, default=1,
                     help="frame stride for evaluation")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "scared":
        from evaluators.scared import SCaredEvaluator
        SCaredEvaluator().run(args)

    elif args.mode == "dtu":
        from evaluators.dtu import DTUEvaluator
        DTUEvaluator().run(args)

    elif args.mode == "bonn":
        from evaluators.bonn import BonnEvaluator
        BonnEvaluator().run(args)

    elif args.mode == "sintel":
        from evaluators.sintel import SintelEvaluator
        SintelEvaluator().run(args)

    elif args.mode == "kitti":
        from evaluators.kitti import KITTIEvaluator
        KITTIEvaluator().run(args)

    elif args.mode == "sintel_pose":
        # resolve deprecated alias
        if args.gt_cam_dir is None and args.gt_traj_dir is not None:
            args.gt_cam_dir = args.gt_traj_dir
        from evaluators.sintel_pose import SintelPoseEvaluator
        SintelPoseEvaluator().run(args)
