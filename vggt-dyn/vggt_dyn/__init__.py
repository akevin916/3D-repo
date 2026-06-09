"""vggt_dyn — VGGT-Dyn: test-time optimization for dynamic scenes.

Core TTO modules (no fine-tuning utilities; see finetune/ for LoRA).
"""

from vggt_dyn.initializer import VGGTInitializer
from vggt_dyn.optimizer import VGGTDynOptimizer, optimization_step, cosine_schedule
from vggt_dyn.dynamic_mask import (
    compute_ego_flow,
    compute_flow_residual,
    build_dynamic_mask,
    pair_mask_to_frame_mask,
    frame_mask_to_pair_mask,
    get_dynamic_mask_from_optimizer,
)
from vggt_dyn.utils.flow_utils import compute_adjacent_flow
from vggt_dyn.utils.pose_utils import decode_pose_enc, compose_pose
from vggt_dyn.pipeline import run_pipeline

__all__ = [
    # initializer
    "VGGTInitializer",
    # optimizer
    "VGGTDynOptimizer",
    "optimization_step",
    "cosine_schedule",
    # dynamic mask
    "compute_ego_flow",
    "compute_flow_residual",
    "build_dynamic_mask",
    "pair_mask_to_frame_mask",
    "frame_mask_to_pair_mask",
    "get_dynamic_mask_from_optimizer",
    # utils
    "compute_adjacent_flow",
    "decode_pose_enc",
    "compose_pose",
    # pipeline
    "run_pipeline",
]

