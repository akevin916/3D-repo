#!/usr/bin/env python3
"""MonST3R-style fine-tuning for VGGT (no LoRA).

Key changes from LoRA pipeline:
- Freeze most backbone parameters.
- Train heads + last N attention blocks (decoder-like adaptation).
- Datasets are modularized under finetune/datasets/*.py.
"""

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

# ── project root on sys.path ───────────────────────────────────────────────
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_MODULE_DIR)
_VGGT_REPO = os.path.join(_PROJECT_DIR, "vggt")
for _p in (_PROJECT_DIR, _VGGT_REPO, os.path.join(_VGGT_REPO, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import extri_intri_to_pose_encoding
from finetune.datasets import build_dataset


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None


def _parse_csv_list(text: str) -> list:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_kv_csv(text: str) -> dict:
    """Parse `k=v,k2=v2` into dict."""
    out = {}
    if not text:
        return out
    for item in _parse_csv_list(text):
        if "=" not in item:
            raise ValueError(f"Invalid key=value item: {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _clone_args_for_dataset(args: argparse.Namespace, dataset_name: str, root: str) -> argparse.Namespace:
    d = argparse.Namespace(**vars(args))
    d.dataset = dataset_name
    d.root = root
    return d


def _build_loader(args: argparse.Namespace):
    """Build single-dataset loader or mixed-ratio loader."""
    if not args.mix:
        dataset = build_dataset(args)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        log.info("Dataset=%s, clips=%d", args.dataset, len(dataset))
        return loader

    names = _parse_csv_list(args.mix_datasets)
    weights = [float(x) for x in _parse_csv_list(args.mix_weights)]
    if len(names) != len(weights):
        raise ValueError("mix_datasets and mix_weights length mismatch")

    root_map = _parse_kv_csv(args.mix_roots)
    datasets = []
    sizes = []
    for name in names:
        root = root_map.get(name)
        if not root:
            raise ValueError(
                f"Missing root for dataset '{name}'. "
                "Provide --mix_roots like point_odyssey=/path,..."
            )
        ds_args = _clone_args_for_dataset(args, name, root)
        ds = build_dataset(ds_args)
        datasets.append(ds)
        sizes.append(len(ds))

    concat = ConcatDataset(datasets)
    sample_weights = []
    for w, n in zip(weights, sizes):
        if n <= 0:
            raise ValueError("A mixed dataset has zero samples")
        # per-item weight -> dataset-level ratio equals w
        sample_weights.extend([w / n] * n)
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=args.mix_samples_per_epoch,
        replacement=True,
    )
    loader = torch.utils.data.DataLoader(
        concat,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    ratio_str = ", ".join(f"{n}:{w}" for n, w in zip(names, weights))
    size_str = ", ".join(f"{n}={s}" for n, s in zip(names, sizes))
    log.info("Mixed datasets enabled")
    log.info("  ratios   : %s", ratio_str)
    log.info("  sizes    : %s", size_str)
    log.info("  sampled clips/epoch: %d", args.mix_samples_per_epoch)
    return loader


def apply_monst3r_style_freeze(
    model: torch.nn.Module,
    train_last_n_blocks: int = 8,
    unfreeze_heads: bool = True,
) -> list:
    """Freeze backbone like MonST3R and train a small subset.

    MonST3R uses freeze='encoder'. VGGT has no explicit decoder split,
    so we map this strategy to:
    1) freeze everything,
    2) unfreeze last N frame/global blocks,
    3) unfreeze output heads.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    agg = model.aggregator
    if train_last_n_blocks > 0:
        n_frame = len(agg.frame_blocks)
        n_global = len(agg.global_blocks)
        n = min(train_last_n_blocks, n_frame, n_global)
        for blk in agg.frame_blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for blk in agg.global_blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)

    if unfreeze_heads:
        for name in ("depth_head", "point_head", "camera_head"):
            head = getattr(model, name, None)
            if head is not None:
                for p in head.parameters():
                    p.requires_grad_(True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    log.info(
        "[Freeze] trainable=%s / %s (%.2f%%), last_n_blocks=%d",
        f"{n_trainable:,}",
        f"{n_total:,}",
        (100.0 * n_trainable / max(1, n_total)),
        train_last_n_blocks,
    )
    return trainable


def _compute_world_points_from_depth(
    depth: torch.Tensor,       # [S, H, W]
    extrinsics: torch.Tensor,  # [S, 3, 4] world-to-cam
    intrinsics: torch.Tensor,  # [S, 3, 3]
) -> torch.Tensor:
    """Back-project GT depth into GT world points [S, H, W, 3]."""
    S, H, W = depth.shape
    device = depth.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(-1, 3)

    K_inv = torch.linalg.inv(intrinsics)
    rays = torch.einsum("sij,pj->spi", K_inv, grid)
    pts_c = rays * depth.reshape(S, -1, 1)

    R = extrinsics[:, :3, :3]
    t = extrinsics[:, :3, 3]
    R_T = R.transpose(1, 2)
    pts_w = torch.einsum("sij,spj->spi", R_T, pts_c - t.unsqueeze(1))
    return pts_w.reshape(S, H, W, 3)


def monst3r_style_loss(
    predictions: Dict[str, torch.Tensor],
    gt_depth: torch.Tensor,
    gt_extrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
    conf_alpha: float,
    camera_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Conf-weighted regression inspired by MonST3R's ConfLoss(Regr3D)."""
    S, H, W = gt_depth.shape
    device = gt_depth.device

    valid = gt_depth > 0
    gt_pts = _compute_world_points_from_depth(gt_depth, gt_extrinsics, gt_intrinsics)

    # Per-scene scale normalization (MonST3R avg_dis norm_mode):
    # normalize by the average distance of valid GT points to origin so that
    # all datasets (indoor ~1m, outdoor ~100m) contribute equally to the loss.
    if valid.sum() > 0:
        scene_scale = gt_pts[valid].norm(dim=-1).mean().clamp(min=1e-3).detach()
        depth_scale = gt_depth[valid].mean().clamp(min=1e-3).detach()
    else:
        scene_scale = torch.tensor(1.0, device=device)
        depth_scale = torch.tensor(1.0, device=device)

    pred_pts = predictions["world_points"]
    pred_conf = predictions["world_points_conf"].clamp(min=1e-6)
    if pred_pts.dim() == 5:  # [1, S, H, W, 3] -> [S, H, W, 3]
        pred_pts = pred_pts.squeeze(0)
    if pred_conf.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_conf = pred_conf.squeeze(0)

    if valid.sum() > 0:
        point_err = ((pred_pts[valid] - gt_pts[valid]) / scene_scale).norm(dim=-1)
        conf = pred_conf[valid]
        loss_point = (point_err * conf - conf_alpha * conf.log()).mean()
    else:
        loss_point = torch.tensor(0.0, device=device)

    pred_depth = predictions["depth"].squeeze(-1)
    pred_depth_conf = predictions["depth_conf"].clamp(min=1e-6)
    if pred_depth.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_depth = pred_depth.squeeze(0)
    if pred_depth_conf.dim() == 4:  # [1, S, H, W] -> [S, H, W]
        pred_depth_conf = pred_depth_conf.squeeze(0)
    if valid.sum() > 0:
        depth_err = ((pred_depth[valid] - gt_depth[valid]) / depth_scale).abs()
        conf_d = pred_depth_conf[valid]
        loss_depth = (depth_err * conf_d - conf_alpha * conf_d.log()).mean()
    else:
        loss_depth = torch.tensor(0.0, device=device)

    loss_camera = torch.tensor(0.0, device=device)
    if "pose_enc_list" in predictions:
        # Normalize GT translation by the same scene_scale (avg_dis) used for
        # point/depth loss, so gt_enc's T matches the normalized scale of
        # VGGT's predicted pose_enc (otherwise datasets with large absolute
        # translations, e.g. waymo, dominate the camera loss).
        gt_extrinsics_n = gt_extrinsics.clone()
        gt_extrinsics_n[:, :, 3] = gt_extrinsics_n[:, :, 3] / scene_scale
        gt_enc = extri_intri_to_pose_encoding(
            gt_extrinsics_n.unsqueeze(0),
            gt_intrinsics.unsqueeze(0),
            image_size_hw=(H, W),
        )
        for pred_enc in predictions["pose_enc_list"]:
            loss_camera = loss_camera + (pred_enc - gt_enc).abs().mean()
        loss_camera = loss_camera / len(predictions["pose_enc_list"])

    total = loss_point + loss_depth + camera_weight * loss_camera
    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_point": float(loss_point.detach().cpu()),
        "loss_depth": float(loss_depth.detach().cpu()),
        "loss_camera": float(loss_camera.detach().cpu()),
    }


def cosine_lr(base_lr: float, min_lr: float, step: int, total_steps: int) -> float:
    t = step / max(1, total_steps - 1)
    return min_lr + (base_lr - min_lr) * (1.0 + math.cos(math.pi * t)) * 0.5


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    epoch: int | None = None,
    global_step: int | None = None,
    best_epoch_loss: float | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "args": vars(args),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if global_step is not None:
        payload["global_step"] = int(global_step)
    if best_epoch_loss is not None:
        payload["best_epoch_loss"] = float(best_epoch_loss)
    torch.save(payload, path)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    writer = None
    if args.tb:
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging requested but tensorboard is not installed. "
                "Install with: pip install tensorboard"
            )
        run_name = args.tb_run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        tb_dir = os.path.join(args.tb_dir, run_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        log.info("TensorBoard enabled: %s", tb_dir)

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

    trainable_params = apply_monst3r_style_freeze(
        model,
        train_last_n_blocks=args.train_last_n_blocks,
        unfreeze_heads=True,
    )

    loader = _build_loader(args)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

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
            start_epoch,
            global_step,
            args.epochs,
        )
    else:
        start_epoch = 0
        global_step = 0
        best_epoch_loss = float("inf")
        end_epoch = args.epochs

    total_steps = max(1, args.epochs * len(loader))

    history = []
    history_path = os.path.join(args.output, "train_history.json")
    if args.resume and os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                history = loaded
        except Exception:
            log.warning("Failed to load existing train_history.json; starting a new history list")

    color_jitter = None
    if args.color_jitter:
        from torchvision.transforms import ColorJitter
        color_jitter = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

    for epoch in range(start_epoch, end_epoch):
        model.train()
        losses = []

        for batch in loader:
            images = batch["images"].squeeze(0).to(device)
            depths = batch["depths"].squeeze(0).to(device)
            extrinsics = batch["extrinsics"].squeeze(0).to(device)
            intrinsics = batch["intrinsics"].squeeze(0).to(device)

            # Per-frame color jitter (image is already a fixed-size tensor from the dataset).
            if color_jitter is not None:
                images = torch.stack([color_jitter(img) for img in images])

            lr = cosine_lr(args.lr, args.min_lr, global_step, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                preds = model(images)
                total_loss, info = monst3r_style_loss(
                    preds,
                    depths,
                    extrinsics,
                    intrinsics,
                    conf_alpha=args.conf_alpha,
                    camera_weight=args.camera_weight,
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
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            info["step"] = global_step
            info["lr"] = lr
            history.append(info)
            losses.append(info["loss_total"])

            if writer is not None:
                writer.add_scalar("train/loss_total", info["loss_total"], global_step)
                writer.add_scalar("train/loss_point", info["loss_point"], global_step)
                writer.add_scalar("train/loss_depth", info["loss_depth"], global_step)
                writer.add_scalar("train/loss_camera", info["loss_camera"], global_step)
                writer.add_scalar("train/lr", lr, global_step)

            if global_step % args.log_every == 0:
                log.info(
                    "[e%d/%d s%d] total=%.4f point=%.4f depth=%.4f camera=%.4f lr=%.2e",
                    epoch + 1,
                    end_epoch,
                    global_step,
                    info["loss_total"],
                    info["loss_point"],
                    info["loss_depth"],
                    info["loss_camera"],
                    lr,
                )
            global_step += 1

        epoch_loss = float(np.mean(losses)) if losses else 0.0
        log.info("[epoch %d] avg_loss=%.4f", epoch + 1, epoch_loss)
        if writer is not None:
            writer.add_scalar("epoch/avg_loss", epoch_loss, epoch + 1)

        save_checkpoint(
            model,
            os.path.join(args.output, f"epoch_{epoch+1:03d}.pt"),
            args,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_epoch_loss=best_epoch_loss,
        )
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            save_checkpoint(
                model,
                os.path.join(args.output, "best.pt"),
                args,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_epoch_loss=best_epoch_loss,
            )

    save_checkpoint(
        model,
        os.path.join(args.output, "final.pt"),
        args,
        optimizer=optimizer,
        scaler=scaler,
        epoch=end_epoch - 1,
        global_step=global_step,
        best_epoch_loss=best_epoch_loss,
    )
    with open(os.path.join(args.output, "train_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    if writer is not None:
        writer.flush()
        writer.close()
    log.info("Training finished. Output: %s", args.output)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("VGGT MonST3R-style fine-tuning")

    p.add_argument("--dataset", default="point_odyssey", choices=["point_odyssey", "tartanair", "waymo", "spring"])
    p.add_argument("--root", default=None, help="Dataset root path (single-dataset mode)")
    p.add_argument("--split", default="train")
    p.add_argument("--difficulty", default="Hard", help="TartanAir only")
    p.add_argument("--camera_id", type=int, default=1, help="Waymo only")

    # Mixed-ratio sampling (MonST3R-style data mixture)
    p.add_argument("--mix", action="store_true", help="Enable multi-dataset mixed-ratio sampling")
    p.add_argument(
        "--mix_datasets",
        default="point_odyssey,tartanair,spring,waymo",
        help="Comma-separated dataset names",
    )
    p.add_argument(
        "--mix_weights",
        default="10000,5000,1000,4000",
        help="Comma-separated sampling weights matching mix_datasets",
    )
    p.add_argument(
        "--mix_roots",
        default="",
        help=(
            "Dataset roots in key=value CSV, e.g. "
            "point_odyssey=/data/point_odyssey,tartanair=/data/tartanair,"
            "spring=/data/spring,waymo=/data/waymo_processed"
        ),
    )
    p.add_argument(
        "--mix_samples_per_epoch",
        type=int,
        default=20000,
        help="How many sampled clips per epoch in mix mode",
    )

    p.add_argument("--ckpt", default=None, help="Base model checkpoint (used when --resume is not set)")
    p.add_argument("--resume", default=None, help="Full finetune checkpoint to resume (model+optimizer+scaler)")
    p.add_argument("--output", required=True)

    p.add_argument("--clip_len", type=int, default=8)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max_depth", type=float, default=200.0)

    # Augmentation (MonST3R alignment)
    p.add_argument("--color_jitter", action="store_true",
                   help="Apply random ColorJitter (brightness/contrast/saturation/hue) per frame")
    p.add_argument("--aug_crop", type=int, default=0,
                   help="Max extra padding pixels for the dataset's crop-resize-to-fixed step "
                        "(MonST3R aug_crop style; applied per-clip in the dataset)")

    p.add_argument("--train_last_n_blocks", type=int, default=8)
    p.add_argument("--conf_alpha", type=float, default=0.2)
    p.add_argument("--camera_weight", type=float, default=0.5)

    p.add_argument("--epochs", type=int, default=10, help="Training epochs (additional epochs when --resume is set)")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--tb_dir", default="runs/finetune", help="TensorBoard log root directory")
    p.add_argument("--tb_run_name", default=None, help="Optional TensorBoard run name")

    args = p.parse_args()
    if not args.mix and not args.root:
        raise ValueError("Single-dataset mode requires --root")
    if not args.resume and not args.ckpt:
        raise ValueError("Either --ckpt (fresh start) or --resume (continue) must be provided")
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
