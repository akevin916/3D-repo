"""Model freeze strategy, seeding, LR schedule, and checkpoint I/O."""

import argparse
import logging
import math
import os
import random

import numpy as np
import torch

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
