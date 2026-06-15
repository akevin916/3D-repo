"""Training/validation DataLoader construction for finetune/train.py."""

import argparse
import logging
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from finetune.datasets import build_dataset

log = logging.getLogger(__name__)


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


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def build_loader(args: argparse.Namespace):
    """Build single-dataset loader or mixed-ratio loader."""
    if not args.mix:
        dataset = build_dataset(args)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=_seed_worker,
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
        worker_init_fn=_seed_worker,
    )

    ratio_str = ", ".join(f"{n}:{w}" for n, w in zip(names, weights))
    size_str = ", ".join(f"{n}={s}" for n, s in zip(names, sizes))
    log.info("Mixed datasets enabled")
    log.info("  ratios   : %s", ratio_str)
    log.info("  sizes    : %s", size_str)
    log.info("  sampled clips/epoch: %d", args.mix_samples_per_epoch)
    return loader
