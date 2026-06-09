#!/usr/bin/env python3
"""train.py — Fine-tune VGGT on dynamic datasets via LoRA (or head-only).

Supports:
  sintel  — MPI-Sintel (depth .dpt, poses .cam, optional GT flow .flo)
  bonn    — Bonn RGB-D  (depth PNG uint16, poses TUM groundtruth.txt)

Usage:
  # Sintel, LoRA on global_blocks
  python finetune/train.py sintel \\
      --image_dir data/sintel/final --depth_dir data/sintel/depth \\
      --cam_dir   data/sintel/camdata_left \\
      --ckpt      vggt/checkpoints/VGGT-1B.pt \\
      --output    finetune_outputs/sintel_lora

  # Bonn, head-only
  python finetune/train.py bonn \\
      --scene_dir data/bonn/rgbd_bonn_dataset/rgbd_bonn_balloon \\
      --ckpt      vggt/checkpoints/VGGT-1B.pt \\
      --output    finetune_outputs/bonn_heads --lora_mode heads
"""

import os
import sys
import json
import math
import argparse
import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── project root on sys.path ───────────────────────────────────────────────
_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))   # finetune/
_PROJECT_DIR = os.path.dirname(_MODULE_DIR)                  # vggt-dyn/
_VGGT_REPO   = os.path.join(_PROJECT_DIR, "vggt")
for _p in (_PROJECT_DIR, _VGGT_REPO, os.path.join(_VGGT_REPO, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from finetune.lora import inject_lora, save_lora
from finetune.datasets import SintelClipDataset, BonnClipDataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Loss
# ===========================================================================

def _compute_world_points_from_depth(
    depth: torch.Tensor,       # [S, H, W]
    extrinsics: torch.Tensor,  # [S, 3, 4]  world-to-cam
    intrinsics: torch.Tensor,  # [S, 3, 3]
) -> torch.Tensor:
    """Back-project GT depth → GT world points [S, H, W, 3]."""
    S, H, W = depth.shape
    device = depth.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid  = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(-1, 3)
    K_inv = torch.linalg.inv(intrinsics)
    rays  = torch.einsum("sij,pj->spi", K_inv, grid)
    pts_c = rays * depth.reshape(S, -1, 1)

    R   = extrinsics[:, :3, :3]
    t   = extrinsics[:, :3,  3]
    R_T = R.transpose(1, 2)
    pts_w = torch.einsum("sij,spj->spi", R_T, pts_c - t.unsqueeze(1))
    return pts_w.reshape(S, H, W, 3)


def dynamic_aware_loss(
    predictions:   dict,
    images:        torch.Tensor,
    gt_depth:      torch.Tensor,
    gt_extrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
    dynamic_mask:  torch.Tensor,
    depth_weight:  float = 1.0,
    point_weight:  float = 1.0,
    camera_weight: float = 0.5,
    alpha: float = 0.2,
    gamma: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """Multi-task loss with dynamic pixels excluded from depth/point terms."""
    S, H, W = gt_depth.shape
    device  = gt_depth.device

    valid_depth = (gt_depth > 0) & ~dynamic_mask
    gt_pts = _compute_world_points_from_depth(gt_depth, gt_extrinsics, gt_intrinsics)

    pred_depth      = predictions["depth"].squeeze(-1)
    pred_depth_conf = predictions["depth_conf"]
    loss_depth = torch.tensor(0.0, device=device)
    if valid_depth.sum() >= 100:
        diff_d = (pred_depth[valid_depth] - gt_depth[valid_depth]).pow(2)
        c_d = pred_depth_conf[valid_depth].clamp(min=1e-6)
        loss_depth = (gamma * diff_d * c_d - alpha * c_d.log()).mean()

    pred_pts      = predictions["world_points"]
    pred_pts_conf = predictions["world_points_conf"]
    loss_point = torch.tensor(0.0, device=device)
    if valid_depth.sum() >= 100:
        diff_p = (pred_pts[valid_depth] - gt_pts[valid_depth]).norm(dim=-1)
        c_p = pred_pts_conf[valid_depth].clamp(min=1e-6)
        loss_point = (gamma * diff_p * c_p - alpha * c_p.log()).mean()

    loss_camera = torch.tensor(0.0, device=device)
    if "pose_enc_list" in predictions:
        ext_b = gt_extrinsics.unsqueeze(0)
        K_b   = gt_intrinsics.unsqueeze(0)
        gt_enc = extri_intri_to_pose_encoding(ext_b, K_b, image_size_hw=(H, W))
        for pred_enc in predictions["pose_enc_list"]:
            loss_camera = loss_camera + (pred_enc - gt_enc).abs().mean()
        loss_camera = loss_camera / len(predictions["pose_enc_list"])

    total = depth_weight * loss_depth + point_weight * loss_point + camera_weight * loss_camera

    return total, {
        "loss_depth":  float(loss_depth),
        "loss_point":  float(loss_point),
        "loss_camera": float(loss_camera),
        "total":       float(total),
        "n_static":    int(valid_depth.sum()),
        "n_dynamic":   int(dynamic_mask.sum()),
    }


# ===========================================================================
# Training
# ===========================================================================

def cosine_lr(base_lr: float, min_lr: float, step: int, total_steps: int) -> float:
    t = step / max(total_steps - 1, 1)
    return min_lr + (base_lr - min_lr) * (1 + math.cos(math.pi * t)) / 2


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    log.info(f"Loading VGGT from {args.ckpt}")
    model = VGGT()
    ckpt  = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model = model.to(device)

    trainable_params = inject_lora(
        model,
        rank           = args.lora_rank,
        alpha          = args.lora_alpha,
        target_blocks  = args.lora_mode,
        unfreeze_heads = True,
    )
    model.train()

    if args.dataset == "sintel":
        seqs = [s.strip() for s in args.sequences.split(",")] if args.sequences else None
        dataset = SintelClipDataset(
            image_dir  = args.image_dir,
            depth_dir  = args.depth_dir,
            cam_dir    = args.cam_dir,
            flow_dir   = getattr(args, "flow_dir", None),
            sequences  = seqs,
            clip_len   = args.clip_len,
            stride     = args.stride,
            dyn_thresh = args.dyn_thresh,
        )
    else:
        dataset = BonnClipDataset(
            scene_dir = args.scene_dir,
            clip_len  = args.clip_len,
            stride    = args.stride,
        )

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * len(loader)
    global_step = 0
    best_total  = float("inf")
    history     = []
    log.info(f"Training {args.epochs} epochs × {len(loader)} clips = {total_steps} steps")

    for epoch in range(args.epochs):
        epoch_losses = []
        for batch in loader:
            images        = batch["images"      ].squeeze(0).to(device)
            gt_depth      = batch["depths"      ].squeeze(0).to(device)
            gt_extrinsics = batch["extrinsics"  ].squeeze(0).to(device)
            gt_intrinsics = batch["intrinsics"  ].squeeze(0).to(device)
            dynamic_mask  = batch["dynamic_mask"].squeeze(0).to(device)

            lr = cosine_lr(args.lr, args.min_lr, global_step, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                predictions = model(images)

            total_loss, loss_dict = dynamic_aware_loss(
                predictions, images, gt_depth, gt_extrinsics, gt_intrinsics,
                dynamic_mask,
                depth_weight=args.depth_weight, point_weight=args.point_weight,
                camera_weight=args.camera_weight,
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            loss_dict["step"] = global_step
            loss_dict["lr"]   = lr
            epoch_losses.append(loss_dict["total"])
            history.append(loss_dict)

            if global_step % args.log_every == 0:
                log.info(
                    f"[e{epoch+1}/{args.epochs} s{global_step}] "
                    f"total={loss_dict['total']:.4f}  "
                    f"depth={loss_dict['loss_depth']:.4f}  "
                    f"point={loss_dict['loss_point']:.4f}  "
                    f"camera={loss_dict['loss_camera']:.4f}  "
                    f"lr={lr:.2e}"
                )
            global_step += 1

        avg = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        log.info(f"[epoch {epoch+1}] avg_loss={avg:.4f}")

        ckpt_path = os.path.join(args.output, f"epoch_{epoch+1:03d}.pt")
        save_lora(model, ckpt_path)

        if avg < best_total:
            best_total = avg
            save_lora(model, os.path.join(args.output, "best.pt"))
            log.info(f"  ↑ best checkpoint saved (loss={best_total:.4f})")

    save_lora(model, os.path.join(args.output, "final.pt"))
    with open(os.path.join(args.output, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Done. Outputs in {args.output}")


# ===========================================================================
# CLI
# ===========================================================================

def _add_common(p):
    p.add_argument("--ckpt",           required=True)
    p.add_argument("--output",         required=True)
    p.add_argument("--lora_mode",      default="global",
                   choices=["global", "frame", "both", "heads"])
    p.add_argument("--lora_rank",      type=int,   default=16)
    p.add_argument("--lora_alpha",     type=float, default=16.0)
    p.add_argument("--epochs",         type=int,   default=3)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--min_lr",         type=float, default=1e-6)
    p.add_argument("--weight_decay",   type=float, default=1e-2)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--clip_len",       type=int,   default=8)
    p.add_argument("--stride",         type=int,   default=4)
    p.add_argument("--depth_weight",   type=float, default=1.0)
    p.add_argument("--point_weight",   type=float, default=1.0)
    p.add_argument("--camera_weight",  type=float, default=0.5)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--log_every",      type=int,   default=10)
    p.add_argument("--amp",            action="store_true")


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT dynamic fine-tuning")
    sub    = parser.add_subparsers(dest="dataset", required=True)

    sp = sub.add_parser("sintel")
    _add_common(sp)
    sp.add_argument("--image_dir",  required=True)
    sp.add_argument("--depth_dir",  required=True)
    sp.add_argument("--cam_dir",    required=True)
    sp.add_argument("--flow_dir",   default=None)
    sp.add_argument("--sequences",  default=None)
    sp.add_argument("--dyn_thresh", type=float, default=1.5)

    bp = sub.add_parser("bonn")
    _add_common(bp)
    bp.add_argument("--scene_dir", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
