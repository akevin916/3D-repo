#!/usr/bin/env python3
"""MonST3R-style fine-tuning for VGGT (no LoRA).

Key changes from LoRA pipeline:
- Freeze most backbone parameters.
- Train heads + last N attention blocks (decoder-like adaptation).
- Datasets are modularized under finetune/datasets/*.py.

Module layout:
- finetune/data_loader.py  : train DataLoader construction (single / mixed-ratio)
- finetune/validation.py   : val DataLoader (PointOdyssey test + Sintel final) + validate()
- finetune/pose_metrics.py : Sim(3) ATE / RPE_rot metrics
- finetune/losses.py       : monst3r_style_loss
- finetune/model_utils.py  : freeze strategy, seeding, cosine LR, checkpoint I/O
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

# ── project root on sys.path ───────────────────────────────────────────────
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_MODULE_DIR)
_VGGT_REPO = os.path.join(_PROJECT_DIR, "vggt")
for _p in (_PROJECT_DIR, _VGGT_REPO, os.path.join(_VGGT_REPO, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vggt.models.vggt import VGGT
from finetune.data_loader import build_loader
from finetune.validation import build_val_loader, validate
from finetune.losses import monst3r_style_loss
from finetune.model_utils import (
    set_seed,
    apply_monst3r_style_freeze,
    cosine_lr,
    save_checkpoint,
    amp_dtype_from_args,
)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from torch.utils.tensorboard import SummaryWriter


def _build_model(args: argparse.Namespace, device: torch.device):
    """Load VGGT (from --ckpt or --resume) and apply the freeze strategy."""
    model = VGGT()
    resume_state = None
    if args.resume:
        log.info("Resuming from checkpoint: %s", args.resume)
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not isinstance(resume_state, dict) or "model" not in resume_state:
            raise ValueError("--resume expects a full finetune checkpoint containing key 'model'")
        model.load_state_dict(resume_state["model"])
    else:
        log.info("Loading VGGT checkpoint: %s", args.ckpt)
        state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    model.to(device)

    param_groups = apply_monst3r_style_freeze(
        model,
        train_last_n_blocks=args.train_last_n_blocks,
        unfreeze_heads=True,
    )
    return model, param_groups, resume_state


def _setup_optimizer(param_groups: dict, args: argparse.Namespace, resume_state):
    """Build optimizer/scaler and resolve epoch/step bookkeeping (incl. resume).

    Two LR groups: backbone (last-N blocks) uses --lr, heads use --lr_heads.
    """
    backbone_params = param_groups.get("backbone", [])
    head_params = [p for name, group in param_groups.items() if name != "backbone" for p in group]

    optimizer_param_groups = []
    if backbone_params:
        optimizer_param_groups.append({"params": backbone_params, "lr": args.lr, "name": "backbone"})
    if head_params:
        optimizer_param_groups.append({"params": head_params, "lr": args.lr_heads, "name": "heads"})

    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.amp_dtype == "float16")

    if resume_state is not None:
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if "scaler" in resume_state and resume_state["scaler"] is not None:
            scaler.load_state_dict(resume_state["scaler"])

        start_epoch = int(resume_state.get("epoch", -1)) + 1
        global_step = int(resume_state.get("global_step", 0))
        best_epoch_loss = float(resume_state.get("best_epoch_loss", float("inf")))
        end_epoch = start_epoch + args.epochs
        log.info(
            "Resume state: start_epoch=%d, global_step=%d, additional_epochs=%d",
            start_epoch, global_step, args.epochs,
        )
    else:
        start_epoch = 0
        global_step = 0
        best_epoch_loss = float("inf")
        end_epoch = args.epochs

    return optimizer, scaler, start_epoch, global_step, best_epoch_loss, end_epoch


def _load_history(args: argparse.Namespace) -> list:
    history_path = os.path.join(args.output, "train_history.json")
    if args.resume and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                return loaded
        except Exception:
            log.warning("Failed to load existing train_history.json; starting a new history list")
    return []


def _run_epoch(model, loader, optimizer, scaler, param_groups, args, device,
               epoch, end_epoch, global_step, total_steps, writer, history):
    """Run one training epoch over `loader`. Returns (epoch_loss, global_step)."""
    model.train()
    losses = []

    amp_dtype = amp_dtype_from_args(args)

    # Base LR of each optimizer param group (backbone vs heads), so the
    # cosine/warmup schedule scales both groups while preserving their ratio.
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    color_jitter = None
    if args.color_jitter:
        from torchvision.transforms import ColorJitter
        color_jitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

    for batch in loader:
        images = batch["images"].squeeze(0).to(device)
        depths = batch["depths"].squeeze(0).to(device)
        extrinsics = batch["extrinsics"].squeeze(0).to(device)
        intrinsics = batch["intrinsics"].squeeze(0).to(device)

        # Per-frame color jitter (image is already a fixed-size tensor from the dataset).
        if color_jitter is not None:
            images = torch.stack([color_jitter(img) for img in images])

        lr = cosine_lr(args.lr, args.min_lr, global_step, total_steps, warmup_ratio=args.warmup_ratio)
        lr_scale = lr / args.lr
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base_lr * lr_scale

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp, dtype=amp_dtype):
            preds = model(images)
            total_loss, info = monst3r_style_loss(
                preds, depths, extrinsics, intrinsics,
                conf_alpha_point=args.conf_alpha_point, conf_alpha_depth=args.conf_alpha_depth,
                point_weight=args.point_weight, depth_weight=args.depth_weight,
                camera_weight=args.camera_weight,
                camera_gamma=args.camera_gamma,
                weight_trans=args.weight_trans, weight_rot=args.weight_rot, weight_focal=args.weight_focal,
                gradient_loss_weight=args.gradient_loss_weight,
                valid_range=args.valid_range,
            )

        if not torch.isfinite(total_loss):
            log.warning("[e%d s%d] non-finite loss=%.4g, skipping step",
                        epoch + 1, global_step, float(total_loss))
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            global_step += 1
            continue

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        # Per-module gradient clipping: clip each group (backbone / depth_head /
        # point_head / camera_head) independently so a large gradient in one
        # task's head doesn't shrink the clip coefficient for the others.
        for group in param_groups.values():
            if group:
                torch.nn.utils.clip_grad_norm_(group, args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        info["step"] = global_step
        info["lr"] = lr
        history.append(info)
        losses.append(info["loss_total"])

        writer.add_scalar("train/loss_total", info["loss_total"], global_step)
        writer.add_scalar("train/loss_point", info["loss_point"], global_step)
        writer.add_scalar("train/loss_depth", info["loss_depth"], global_step)
        writer.add_scalar("train/loss_camera", info["loss_camera"], global_step)
        writer.add_scalar("train/loss_T", info["loss_T"], global_step)
        writer.add_scalar("train/loss_R", info["loss_R"], global_step)
        writer.add_scalar("train/loss_FL", info["loss_FL"], global_step)
        writer.add_scalar("train/loss_grad_point", info["loss_grad_point"], global_step)
        writer.add_scalar("train/loss_grad_depth", info["loss_grad_depth"], global_step)
        writer.add_scalar("train/lr", lr, global_step)

        if global_step % args.log_every == 0:
            log.info(
                "[e%d/%d s%d] total=%.4f point=%.4f depth=%.4f camera=%.4f lr=%.2e",
                epoch + 1, end_epoch, global_step,
                info["loss_total"], info["loss_point"], info["loss_depth"], info["loss_camera"],
                lr,
            )
        global_step += 1

    epoch_loss = float(np.mean(losses)) if losses else 0.0
    return epoch_loss, global_step


def _run_validation(model, val_loader, args, device, epoch, writer):
    """Run validation (if due) and log/report metrics. Returns val_metrics or None."""
    if val_loader is None or (epoch + 1) % args.val_every != 0:
        return None

    val_metrics = validate(model, val_loader, args, device)
    log.info(
        "[epoch %d] val: loss=%.4f abs_rel=%.4f rmse=%.4f d1=%.4f ate=%.4f rpe_rot=%.4f",
        epoch + 1,
        val_metrics.get("loss_total", float("nan")),
        val_metrics.get("abs_rel", float("nan")),
        val_metrics.get("rmse", float("nan")),
        val_metrics.get("d1", float("nan")),
        val_metrics.get("ate", float("nan")),
        val_metrics.get("rpe_rot", float("nan")),
    )
    for k, v in val_metrics.items():
        writer.add_scalar(f"val/{k}", v, epoch + 1)
    return val_metrics


def _save_epoch_checkpoints(model, optimizer, scaler, args, epoch, global_step,
                             best_epoch_loss, epoch_loss, val_metrics):
    """Save the per-epoch checkpoint, and best.pt if a new best was reached.

    Returns the (possibly updated) best_epoch_loss.
    """
    save_checkpoint(
        model,
        os.path.join(args.output, f"epoch_{epoch+1:03d}.pt"),
        args,
        optimizer=optimizer, scaler=scaler,
        epoch=epoch, global_step=global_step, best_epoch_loss=best_epoch_loss,
    )

    compare_loss = val_metrics["loss_total"] if val_metrics is not None else epoch_loss
    if compare_loss < best_epoch_loss:
        log.info(
            "[epoch %d] New best %s=%.4f (prev=%.4f)",
            epoch + 1,
            "val_loss" if val_metrics is not None else "train_loss",
            compare_loss, best_epoch_loss,
        )
        best_epoch_loss = compare_loss
        save_checkpoint(
            model,
            os.path.join(args.output, "best.pt"),
            args,
            optimizer=optimizer, scaler=scaler,
            epoch=epoch, global_step=global_step, best_epoch_loss=best_epoch_loss,
        )

    return best_epoch_loss


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    set_seed(args.seed)
    log.info("Seed: %d", args.seed)

    with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    tb_dir = os.path.join(args.output, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    log.info("TensorBoard enabled: %s", tb_dir)

    model, param_groups, resume_state = _build_model(args, device)

    loader = build_loader(args)
    val_loader = build_val_loader(args)

    optimizer, scaler, start_epoch, global_step, best_epoch_loss, end_epoch = _setup_optimizer(
        param_groups, args, resume_state
    )
    total_steps = max(1, args.epochs * len(loader))
    history = _load_history(args)

    train_start = time.time()
    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()

        epoch_loss, global_step = _run_epoch(
            model, loader, optimizer, scaler, param_groups, args, device,
            epoch, end_epoch, global_step, total_steps, writer, history,
        )

        epoch_time = time.time() - epoch_start
        log.info("[epoch %d] avg_loss=%.4f time=%.1fs", epoch + 1, epoch_loss, epoch_time)
        writer.add_scalar("epoch/avg_loss", epoch_loss, epoch + 1)
        writer.add_scalar("epoch/time_sec", epoch_time, epoch + 1)

        val_metrics = _run_validation(model, val_loader, args, device, epoch, writer)

        best_epoch_loss = _save_epoch_checkpoints(
            model, optimizer, scaler, args, epoch, global_step,
            best_epoch_loss, epoch_loss, val_metrics,
        )

    save_checkpoint(
        model,
        os.path.join(args.output, "final.pt"),
        args,
        optimizer=optimizer, scaler=scaler,
        epoch=end_epoch - 1, global_step=global_step, best_epoch_loss=best_epoch_loss,
    )
    with open(os.path.join(args.output, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    writer.flush()
    writer.close()
    total_time = time.time() - train_start
    log.info("Training finished in %.1fs. Output: %s", total_time, args.output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("VGGT MonST3R-style fine-tuning")

    g_data = p.add_argument_group("Dataset (single-dataset mode)")
    g_data.add_argument("--dataset", default="point_odyssey", choices=["point_odyssey", "tartanair", "waymo", "spring"])
    g_data.add_argument("--root", default=None, help="Dataset root path (single-dataset mode)")
    g_data.add_argument("--split", default="train")
    g_data.add_argument("--difficulty", default="Hard", help="TartanAir only")
    g_data.add_argument("--camera_id", type=int, default=1, help="Waymo only")

    g_mix = p.add_argument_group("Mixed-ratio sampling (MonST3R-style data mixture)")
    g_mix.add_argument("--mix", action="store_true", help="Enable multi-dataset mixed-ratio sampling")
    g_mix.add_argument(
        "--mix_datasets",
        default="point_odyssey,tartanair,spring,waymo",
        help="Comma-separated dataset names",
    )
    g_mix.add_argument(
        "--mix_weights",
        default="10000,5000,1000,4000",
        help="Comma-separated sampling weights matching mix_datasets",
    )
    g_mix.add_argument(
        "--mix_roots",
        default="",
        help=(
            "Dataset roots in key=value CSV, e.g. "
            "point_odyssey=/data/point_odyssey,tartanair=/data/tartanair,"
            "spring=/data/spring,waymo=/data/waymo_processed"
        ),
    )
    g_mix.add_argument(
        "--mix_samples_per_epoch",
        type=int,
        default=20000,
        help="How many sampled clips per epoch in mix mode",
    )

    g_ckpt = p.add_argument_group("Checkpoints")
    g_ckpt.add_argument("--ckpt", default=None, help="Base model checkpoint (used when --resume is not set)")
    g_ckpt.add_argument("--resume", default=None, help="Full finetune checkpoint to resume (model+optimizer+scaler)")
    g_ckpt.add_argument("--output", required=True)

    g_clip = p.add_argument_group("Clip sampling")
    g_clip.add_argument("--clip_len", type=int, default=8)
    g_clip.add_argument("--stride", type=int, default=4)
    g_clip.add_argument("--max_depth", type=float, default=200.0)

    g_aug = p.add_argument_group("Augmentation (MonST3R alignment)")
    g_aug.add_argument("--color_jitter", action="store_true",
                        help="Apply random ColorJitter (brightness/contrast/saturation/hue) per frame")
    g_aug.add_argument("--aug_crop", type=int, default=0,
                        help="Max extra padding pixels for the dataset's crop-resize-to-fixed step "
                             "(MonST3R aug_crop style; applied per-clip in the dataset)")

    g_train = p.add_argument_group("Training / optimization")
    g_train.add_argument("--train_last_n_blocks", type=int, default=8)
    g_train.add_argument("--conf_alpha_point", type=float, default=0.2, help="conf regularization weight for point loss")
    g_train.add_argument("--conf_alpha_depth", type=float, default=0.2, help="conf regularization weight for depth loss")
    g_train.add_argument("--point_weight", type=float, default=1.0, help="Overall weight for the point loss term")
    g_train.add_argument("--depth_weight", type=float, default=1.0, help="Overall weight for the depth loss term")
    g_train.add_argument("--camera_weight", type=float, default=0.5)
    g_train.add_argument("--camera_gamma", type=float, default=0.6,
                          help="Temporal decay for multi-stage camera loss (gamma^(n_stages-stage-1))")
    g_train.add_argument("--weight_trans", type=float, default=1.0, help="Camera loss weight for translation")
    g_train.add_argument("--weight_rot", type=float, default=1.0, help="Camera loss weight for rotation")
    g_train.add_argument("--weight_focal", type=float, default=0.5, help="Camera loss weight for focal/FoV")
    g_train.add_argument("--gradient_loss_weight", type=float, default=0.0,
                          help="Weight for spatial-gradient consistency loss on depth/point maps (0 = disabled)")
    g_train.add_argument("--valid_range", type=float, default=0.98,
                          help="Quantile threshold for outlier-pixel filtering in regression loss")
    g_train.add_argument("--epochs", type=int, default=10, help="Training epochs (additional epochs when --resume is set)")
    g_train.add_argument("--lr", type=float, default=5e-5, help="Backbone (last-N blocks) learning rate")
    g_train.add_argument("--lr_heads", type=float, default=5e-5, help="Output heads (depth/point/camera) learning rate")
    g_train.add_argument("--min_lr", type=float, default=1e-6)
    g_train.add_argument("--weight_decay", type=float, default=0.05)
    g_train.add_argument("--grad_clip", type=float, default=1.0)
    g_train.add_argument("--warmup_ratio", type=float, default=0.05, help="Fraction of total steps for linear LR warmup")
    g_train.add_argument("--seed", type=int, default=42, help="Random seed for python/numpy/torch")

    g_val = p.add_argument_group("Validation (MonST3R-style: PointOdyssey test split + Sintel 'final')")
    g_val.add_argument("--val_split", default="test", help="PointOdyssey split name for validation data")
    g_val.add_argument("--val_every", type=int, default=1, help="Run validation every N epochs")
    g_val.add_argument("--sintel_root", default=None,
                        help="Path to Sintel root (containing training/), used for validation")

    g_run = p.add_argument_group("Runtime")
    g_run.add_argument("--workers", type=int, default=4)
    g_run.add_argument("--device", default="cuda")
    g_run.add_argument("--amp", action="store_true")
    g_run.add_argument("--amp_dtype", default="bfloat16", choices=["bfloat16", "float16"],
                        help="Autocast dtype when --amp is set (bfloat16 needs no GradScaler)")
    g_run.add_argument("--log_every", type=int, default=10)

    args = p.parse_args()
    if not args.mix and not args.root:
        raise ValueError("Single-dataset mode requires --root")
    if not args.resume and not args.ckpt:
        raise ValueError("Either --ckpt (fresh start) or --resume (continue) must be provided")
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
