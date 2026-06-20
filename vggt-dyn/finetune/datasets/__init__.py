"""Dataset package for MonST3R-style VGGT fine-tuning."""

from .common import DATASET_TARGET_HW
from .point_odyssey import PointOdysseyClipDataset
from .tartanair import TartanAirClipDataset
from .waymo import WaymoClipDataset
from .spring import SpringClipDataset
from .sintel import SintelClipDataset


def build_dataset(args):
    name = args.dataset
    target_hw = DATASET_TARGET_HW[name]
    aug_crop = getattr(args, "aug_crop", 0)
    expand_ratio = getattr(args, "expand_ratio", 0.0)
    stride = getattr(args, "stride", 4)
    if name == "point_odyssey":
        return PointOdysseyClipDataset(
            root=args.root,
            split=args.split,
            clip_len=args.clip_len,
            stride=stride,
            expand_ratio=expand_ratio,
            max_depth=args.max_depth,
            target_hw=target_hw,
            aug_crop=aug_crop,
        )
    if name == "tartanair":
        return TartanAirClipDataset(
            root=args.root,
            split=args.split,
            difficulty=args.difficulty,
            clip_len=args.clip_len,
            stride=stride,
            expand_ratio=expand_ratio,
            max_depth=args.max_depth,
            target_hw=target_hw,
            aug_crop=aug_crop,
        )
    if name == "waymo":
        return WaymoClipDataset(
            root=args.root,
            clip_len=args.clip_len,
            stride=stride,
            expand_ratio=expand_ratio,
            camera_id=args.camera_id,
            max_depth=args.max_depth,
            target_hw=target_hw,
            aug_crop=aug_crop,
        )
    if name == "spring":
        return SpringClipDataset(
            root=args.root,
            split=args.split,
            clip_len=args.clip_len,
            stride=stride,
            expand_ratio=expand_ratio,
            max_depth=args.max_depth,
            target_hw=target_hw,
            aug_crop=aug_crop,
        )
    if name == "sintel":
        return SintelClipDataset(
            root=args.root,
            split=args.split,
            clip_len=args.clip_len,
            stride=stride,
            expand_ratio=expand_ratio,
            max_depth=args.max_depth,
            target_hw=target_hw,
            aug_crop=aug_crop,
        )
    raise ValueError(f"Unsupported dataset: {name}")
