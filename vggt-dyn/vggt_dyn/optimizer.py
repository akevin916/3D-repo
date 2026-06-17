"""
optimizer.py — VGGTDynOptimizer: test-time pose/depth refinement for dynamic scenes.

Design:
  • No pair graph — VGGT gives per-frame world_points directly in world coordinates.
  • Optimization variables: delta_depth (log-space), delta_rotvec, delta_t.
  • Loss computation is fully delegated to an injected nn.Module (MonLoss / DynLoss).
    See losses.py for available strategies.
  • dynamic_mask is a buffer updated externally every N iterations.
"""

import os
import sys
import math

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from vggt_dyn.utils.pose_utils import compose_pose


# ---------------------------------------------------------------------------
# Learning-rate schedule helpers
# ---------------------------------------------------------------------------

def cosine_schedule(t: float, lr_base: float, lr_min: float) -> float:
    return lr_min + (lr_base - lr_min) * (1.0 + math.cos(math.pi * t)) / 2.0


def optimization_step(
    net: "VGGTDynOptimizer",
    cur_iter: int,
    niter: int,
    lr_base: float,
    lr_min: float,
    optimizer: torch.optim.Optimizer,
) -> tuple:
    """Single gradient-descent step (mirrors MonST3R's global_alignment_iter).

    Returns:
        (total_loss, secondary_loss, current_lr)
    """
    t  = cur_iter / niter
    lr = cosine_schedule(t, lr_base, lr_min)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad()
    total_loss, secondary_loss = net(epoch=cur_iter)
    total_loss.backward()
    optimizer.step()

    return float(total_loss), float(secondary_loss), lr


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class VGGTDynOptimizer(nn.Module):
    """Test-time optimizer that refines VGGT pose/depth predictions for
    dynamic scenes.

    Manages optimization variables (delta_depth, delta_rotvec, delta_t) and
    frozen VGGT buffers.  All loss computation is handled by the injected
    ``loss`` module, keeping this class focused on geometry and parameter
    management.

    Args:
        init        : VGGTInitializer with VGGT outputs already parsed
        flow_fwd    : [S-1, 2, H, W] RAFT forward flow  (t → t+1)
        valid_fwd   : [S-1, 1, H, W] bool — valid flow mask
        loss        : nn.Module with signature forward(opt, epoch) → (total, secondary)
                      Use MonLoss or DynLoss from vggt_dyn.losses.
        freeze_pose : if True, delta_rotvec and delta_t are frozen (depth-only opt)
    """

    def __init__(
        self,
        init,
        flow_fwd: torch.Tensor,
        valid_fwd: torch.Tensor,
        loss: nn.Module,
        freeze_pose: bool = False,
    ):
        super().__init__()

        S, H, W = init.S, init.H, init.W
        self.S = S
        self.H = H
        self.W = W

        # ---- optimization variables (initialized to zero = identity delta) ----
        self.delta_depth  = nn.Parameter(torch.zeros(S, H * W))
        self.delta_rotvec = nn.Parameter(torch.zeros(S, 3), requires_grad=not freeze_pose)
        self.delta_t      = nn.Parameter(torch.zeros(S, 3), requires_grad=not freeze_pose)

        # ---- frozen VGGT initialization (buffers, not parameters) ----
        self.register_buffer("init_depth",   init.depth)       # [S, H, W]
        self.register_buffer("init_R",       init.R)           # [S, 3, 3]
        self.register_buffer("init_T",       init.T)           # [S, 3]
        self.register_buffer("K",            init.K)           # [S, 3, 3]
        self.register_buffer("anchor_pts",   init.anchor_pts)  # [S, H, W, 3]
        self.register_buffer("anchor_conf",  init.anchor_conf) # [S, H, W]

        # ---- precomputed RAFT flow (frozen) ----
        self.register_buffer("flow_fwd",  flow_fwd)
        self.register_buffer("valid_fwd", valid_fwd.bool())

        # ---- dynamic mask (updated every N iters from outside) ----
        self.register_buffer(
            "dynamic_mask",
            torch.zeros(S - 1, 1, H, W, dtype=torch.bool),
        )

        # ---- loss strategy ----
        self.loss = loss

    # ------------------------------------------------------------------
    # Refined state
    # ------------------------------------------------------------------

    def get_refined_depth(self) -> torch.Tensor:
        """[S, H, W] depth = init_depth * exp(delta_depth)"""
        return self.init_depth * torch.exp(
            self.delta_depth.view(self.S, self.H, self.W)
        )

    def get_refined_RT(self):
        """([S,3,3], [S,3]) — refined rotation and translation."""
        return compose_pose(self.init_R, self.init_T,
                            self.delta_rotvec, self.delta_t)

    def _depth_to_world(
        self,
        depth: torch.Tensor,   # [S, H, W]
        R: torch.Tensor,       # [S, 3, 3] cam-from-world
        T: torch.Tensor,       # [S, 3]
    ) -> torch.Tensor:         # [S, H, W, 3]
        """Backproject depth → world-frame 3D points.

        x_w = R^T @ (K_inv @ [u,v,1]^T * depth - T)
        """
        S, H, W = depth.shape
        device  = depth.device

        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)  # [H, W, 3]
        grid = grid.reshape(-1, 3)                                  # [H*W, 3]

        K_inv = torch.linalg.inv(self.K)                           # [S, 3, 3]
        rays  = torch.einsum("sij,pj->spi", K_inv, grid)           # [S, H*W, 3]
        pts_c = rays * depth.reshape(S, H * W, 1)                  # [S, H*W, 3]

        R_t           = R.transpose(1, 2)                          # [S, 3, 3]
        pts_c_shifted = pts_c - T.unsqueeze(1)                     # [S, H*W, 3]
        pts_w         = torch.einsum("sij,spj->spi", R_t, pts_c_shifted)

        return pts_w.reshape(S, H, W, 3)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, epoch: int = 0):
        """Delegate to the injected loss strategy.

        Returns:
            (total_loss, secondary_loss) — both scalar tensors
        """
        return self.loss(self, epoch)

    # ------------------------------------------------------------------
    # External updates
    # ------------------------------------------------------------------

    def update_dynamic_mask(self, mask: torch.Tensor):
        """Replace dynamic_mask buffer.

        Args:
            mask: [S-1, 1, H, W] bool tensor (True = dynamic pixel)
        """
        self.dynamic_mask.copy_(mask.bool())

    # ------------------------------------------------------------------
    # Result extraction (no_grad)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_pts3d(self) -> torch.Tensor:
        """[S, H, W, 3] refined world-frame 3D points."""
        depth = self.get_refined_depth()
        R, T  = self.get_refined_RT()
        return self._depth_to_world(depth, R, T)

    @torch.no_grad()
    def get_depth(self) -> torch.Tensor:
        """[S, H, W] refined depth maps."""
        return self.get_refined_depth()

    @torch.no_grad()
    def get_extrinsics(self) -> torch.Tensor:
        """[S, 3, 4] refined camera extrinsics (cam-from-world)."""
        R, T = self.get_refined_RT()
        return torch.cat([R, T.unsqueeze(-1)], dim=-1)

    @torch.no_grad()
    def get_K(self) -> torch.Tensor:
        """[S, 3, 3] intrinsics (fixed, from VGGT)."""
        return self.K.clone()
