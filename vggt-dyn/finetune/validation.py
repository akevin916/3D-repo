"""MonST3R-style validation: PointOdyssey(test) + Sintel(final) loaders and metrics."""

import argparse
import logging
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from finetune.datasets import build_dataset
from finetune.data_loader import _clone_args_for_dataset, _parse_kv_csv
from finetune.losses import monst3r_style_loss
from finetune.pose_metrics import w2c_to_c2w, pose_metrics

log = logging.getLogger(__name__)


def _add_val_dataset(datasets: list, args: argparse.Namespace, name: str, root: str, split: str) -> None:
    val_args = _clone_args_for_dataset(args, name, root)
    val_args.split = split
    try:
        ds = build_dataset(val_args)
    except FileNotFoundError as e:
        log.warning("[val] %s; skipping", e)
        return
    if len(ds) == 0:
        log.warning("[val] dataset=%s split=%s is empty; skipping", name, split)
        return
    datasets.append(ds)
    log.info("[val] dataset=%s split=%s clips=%d", name, split, len(ds))


def build_val_loader(args: argparse.Namespace):
    """Build a MonST3R-style validation loader: PointOdyssey(split=val_split)
    + Sintel(split='final'), independent of the training dataset(s).

    PointOdyssey root is taken from --root (single mode) or --mix_roots
    (mix mode); Sintel root is taken from --sintel_root. Either source is
    skipped (with a warning) if its data is unavailable.
    """
    datasets: list = []

    po_root = None
    if not args.mix:
        if args.dataset == "point_odyssey":
            po_root = args.root
    else:
        po_root = _parse_kv_csv(args.mix_roots).get("point_odyssey")

    if po_root:
        _add_val_dataset(datasets, args, "point_odyssey", po_root, args.val_split)
    else:
        log.info("[val] no point_odyssey root available; skipping PointOdyssey validation")

    if args.sintel_root:
        _add_val_dataset(datasets, args, "sintel", args.sintel_root, "final")
    else:
        log.info("[val] --sintel_root not set; skipping Sintel validation")

    if not datasets:
        log.warning("[val] no validation datasets available; validation disabled")
        return None

    concat = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(
        concat,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    log.info("[val] total clips=%d", len(concat))
    return loader


@torch.no_grad()
def validate(model: torch.nn.Module, val_loader, args: argparse.Namespace, device: torch.device) -> Optional[Dict[str, float]]:
    """Lightweight validation: regression loss + depth/pose metrics, mirroring evaluators/."""
    if val_loader is None:
        return None

    model.eval()
    agg: Dict[str, list] = {
        "loss_total": [], "loss_point": [], "loss_depth": [], "loss_camera": [],
        "abs_rel": [], "rmse": [], "d1": [],
        "ate": [], "rpe_rot": [],
    }

    for batch in val_loader:
        images = batch["images"].squeeze(0).to(device)
        depths = batch["depths"].squeeze(0).to(device)
        extrinsics = batch["extrinsics"].squeeze(0).to(device)
        intrinsics = batch["intrinsics"].squeeze(0).to(device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            preds = model(images)
            _, info = monst3r_style_loss(
                preds, depths, extrinsics, intrinsics,
                conf_alpha=args.conf_alpha, camera_weight=args.camera_weight,
            )
        for k in ("loss_total", "loss_point", "loss_depth", "loss_camera"):
            agg[k].append(info[k])

        # ---- depth metrics (median-scale aligned, like evaluators/metrics.py) ----
        pred_depth = preds["depth"].squeeze(-1)
        if pred_depth.dim() == 4:  # [1, S, H, W] -> [S, H, W]
            pred_depth = pred_depth.squeeze(0)
        valid = depths > 0
        if valid.sum() > 0:
            pred_np = pred_depth[valid].detach().float().cpu().numpy()
            gt_np = depths[valid].detach().float().cpu().numpy()
            scale = np.median(gt_np) / (np.median(pred_np) + 1e-8)
            pred_aligned = pred_np * scale
            agg["abs_rel"].append(float(np.mean(np.abs(pred_aligned - gt_np) / gt_np)))
            agg["rmse"].append(float(np.sqrt(np.mean((pred_aligned - gt_np) ** 2))))
            thresh = np.maximum(pred_aligned / gt_np, gt_np / pred_aligned)
            agg["d1"].append(float(np.mean(thresh < 1.25)))

        # ---- pose metrics (Sim3-aligned ATE / RPE_rot, like evaluators/sintel_pose.py) ----
        if "pose_enc_list" in preds and extrinsics.shape[0] >= 2:
            S, H, W = depths.shape
            pred_enc = preds["pose_enc_list"][-1]
            pred_extri, _ = pose_encoding_to_extri_intri(
                pred_enc, image_size_hw=(H, W), build_intrinsics=False
            )
            pred_extri = pred_extri.squeeze(0).detach().float().cpu().numpy()  # [S, 3, 4] w2c
            gt_extri = extrinsics.detach().float().cpu().numpy()  # [S, 3, 4] w2c

            pred_c2w = w2c_to_c2w(pred_extri)
            gt_c2w = w2c_to_c2w(gt_extri)

            ate, rpe_rot = pose_metrics(gt_c2w, pred_c2w)
            agg["ate"].append(ate)
            agg["rpe_rot"].append(rpe_rot)

    model.train()
    return {k: float(np.mean(v)) for k, v in agg.items() if v}
