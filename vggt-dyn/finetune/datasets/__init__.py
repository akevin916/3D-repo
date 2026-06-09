"""Dataset package for MonST3R-style VGGT fine-tuning."""

from .point_odyssey import PointOdysseyClipDataset
from .tartanair import TartanAirClipDataset
from .waymo import WaymoClipDataset
from .spring import SpringClipDataset


def build_dataset(args):
    name = args.dataset
    if name == "point_odyssey":
        return PointOdysseyClipDataset(
            root=args.root,
            split=args.split,
            clip_len=args.clip_len,
            stride=args.stride,
            max_depth=args.max_depth,
        )
    if name == "tartanair":
        return TartanAirClipDataset(
            root=args.root,
            split=args.split,
            difficulty=args.difficulty,
            clip_len=args.clip_len,
            stride=args.stride,
        )
    if name == "waymo":
        return WaymoClipDataset(
            root=args.root,
            clip_len=args.clip_len,
            stride=args.stride,
            camera_id=args.camera_id,
            max_depth=args.max_depth,
        )
    if name == "spring":
        return SpringClipDataset(
            root=args.root,
            split=args.split,
            clip_len=args.clip_len,
            stride=args.stride,
            max_depth=args.max_depth,
        )
    raise ValueError(f"Unsupported dataset: {name}")
