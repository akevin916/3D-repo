"""
optimizer.py — VGGTDynOptimizer: test-time pose/depth refinement for dynamic scenes.

Design:
  • No pair graph — VGGT gives per-frame world_points directly in world coordinates.
  • Optimization variables: delta_depth (log-space), delta_rotvec, delta_t.
  • Losses:
      1. Anchor loss    — refined world pts vs frozen VGGT world_points (weighted by conf)
      2. Flow loss      — ego-motion flow vs RAFT flow on static pixels only
      3. Depth reg loss — scale-invariant regularization toward VGGT init depth
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
import torch.nn.functional as F

from vggt_dyn.utils.pose_utils import compose_pose
from third_party.goem_opt import DepthBasedWarping, depth_regularization_si_weighted


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
        (total_loss, flow_loss, current_lr)
    """
    t  = cur_iter / niter
    lr = cosine_schedule(t, lr_base, lr_min)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    optimizer.zero_grad()
    total_loss, flow_loss = net(epoch=cur_iter)
    total_loss.backward()
    optimizer.step()

    return float(total_loss), float(flow_loss), lr


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class VGGTDynOptimizer(nn.Module):
    """Test-time optimizer that refines VGGT pose/depth predictions for
    dynamic scenes using ego-flow consistency and VGGT anchor losses.

    Args:
        init               : VGGTInitializer with VGGT outputs already parsed
        flow_fwd           : [S-1, 2, H, W] RAFT forward flow  (t → t+1)
        valid_fwd          : [S-1, 1, H, W] bool — valid flow mask (fwd)
        anchor_weight      : (mon) weight for anchor loss
        flow_weight        : (mon) weight for flow loss
        depth_reg_weight   : (mon) weight for depth regularization
        mon_smooth_weight  : (mon) weight for global 3D track smoothness (all pixels)
        dyn_pointmap_weight: (dyn) λ — weight for dynamic pointmap loss
        track_smooth_weight: (dyn) γ — weight for 3D track smoothness (dynamic pixels only)
        loss_version       : 'mon' (default) or 'dyn'
    """

    def __init__(
        self,
        init,
        flow_fwd: torch.Tensor,
        valid_fwd: torch.Tensor,
        anchor_weight: float       = 1.0,
        flow_weight: float         = 1.0,
        depth_reg_weight: float    = 0.1,
        mon_smooth_weight: float   = 0.1,
        dyn_pointmap_weight: float = 1.0,
        track_smooth_weight: float = 0.1,
        loss_version: str          = "mon",
        freeze_pose: bool          = False,
    ):
        super().__init__()

        S, H, W = init.S, init.H, init.W
        self.S = S
        self.H = H
        self.W = W

        # ---- optimization variables (initialized to zero = identity delta) ----
        self.delta_depth   = nn.Parameter(torch.zeros(S, H * W))  # log-space
        self.delta_rotvec  = nn.Parameter(torch.zeros(S, 3), requires_grad=not freeze_pose)  # axis-angle
        self.delta_t       = nn.Parameter(torch.zeros(S, 3), requires_grad=not freeze_pose)

        # ---- frozen VGGT initialization (buffers, not parameters) ----
        self.register_buffer("init_depth",   init.depth)       # [S, H, W]
        self.register_buffer("init_R",       init.R)           # [S, 3, 3]
        self.register_buffer("init_T",       init.T)           # [S, 3]
        self.register_buffer("K",            init.K)           # [S, 3, 3]
        self.register_buffer("anchor_pts",   init.anchor_pts)  # [S, H, W, 3]
        self.register_buffer("anchor_conf",  init.anchor_conf) # [S, H, W]

        # ---- precomputed RAFT flow (frozen) ----
        self.register_buffer("flow_fwd",  flow_fwd)    # [S-1, 2, H, W]
        self.register_buffer("valid_fwd", valid_fwd.bool())   # [S-1, 1, H, W]

        # ---- dynamic mask (updated every N iters from outside) ----
        self.register_buffer(
            "dynamic_mask",
            torch.zeros(S - 1, 1, H, W, dtype=torch.bool),
        )

        # ---- loss weights ----
        self.anchor_weight       = anchor_weight
        self.flow_weight         = flow_weight
        self.depth_reg_weight    = depth_reg_weight
        self.mon_smooth_weight   = mon_smooth_weight
        self.dyn_pointmap_weight = dyn_pointmap_weight
        self.track_smooth_weight = track_smooth_weight
        self.loss_version        = loss_version

        # ---- geometry helpers ----
        self._depth_warping = DepthBasedWarping()

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

        # world = R^T @ (pts_c - T)
        R_t   = R.transpose(1, 2)                                  # [S, 3, 3]
        pts_c_shifted = pts_c - T.unsqueeze(1)                     # [S, H*W, 3]
        pts_w = torch.einsum("sij,spj->spi", R_t, pts_c_shifted)   # [S, H*W, 3]

        return pts_w.reshape(S, H, W, 3)

    # ------------------------------------------------------------------
    # mon losses
    # ------------------------------------------------------------------

    def _anchor_loss(self, refined_pts: torch.Tensor) -> torch.Tensor:
        """Anchor loss: weighted L2 between refined world pts and VGGT anchor.

        Loss = mean(anchor_conf * ||refined_pts - anchor_pts||^2)
        """
        diff    = refined_pts - self.anchor_pts            # [S, H, W, 3]
        sq_dist = (diff ** 2).sum(dim=-1)                  # [S, H, W]
        return (self.anchor_conf * sq_dist).mean()

    def _flow_loss(
        self,
        depth: torch.Tensor,   # [S, H, W]
        R: torch.Tensor,       # [S, 3, 3]
        T: torch.Tensor,       # [S, 3]
    ) -> torch.Tensor:
        """Ego-flow consistency loss on static pixels only.

        Computes ego-motion optical flow for adjacent (t, t+1) pairs using
        DepthBasedWarping, then compares with precomputed RAFT flow.
        """
        if self.S < 2:
            return torch.zeros(1, device=depth.device).squeeze()

        K     = self.K[:-1]                            # [S-1, 3, 3]
        K_inv = torch.linalg.inv(K)                    # [S-1, 3, 3]

        # R, T are cam-from-world (X_cam = R @ X_world + T); DepthBasedWarping
        # expects cam-to-world (X_world = R_c2w @ X_cam + t_c2w).
        R_c2w = R.transpose(-1, -2)                          # [S, 3, 3]
        t_c2w = -(R_c2w @ T.unsqueeze(-1))                   # [S, 3, 1]

        # DepthBasedWarping signature:
        # forward(src_R, src_t, tgt_R, tgt_t, src_disp, K, inv_K, use_depth=False)
        # src_t / tgt_t must be [..., 3, 1] (column vectors)
        ego_hom, _ = self._depth_warping(
            R_c2w[:-1],                # [S-1, 3, 3]
            t_c2w[:-1],                # [S-1, 3, 1]
            R_c2w[1:],                 # [S-1, 3, 3]
            t_c2w[1:],                 # [S-1, 3, 1]
            depth[:-1].unsqueeze(1),   # [S-1, 1, H, W]
            K,                         # [S-1, 3, 3]
            K_inv,                     # [S-1, 3, 3]
            use_depth=True,
        )
        # ego_hom: [S-1, 3, H, W]; channels 0-1 are pixel-space flow (x, y)
        ego_flow = ego_hom[:, :2]      # [S-1, 2, H, W]

        # static mask: valid RAFT flow AND not detected as dynamic
        static_mask = self.valid_fwd & ~self.dynamic_mask   # [S-1, 1, H, W]
        mask_2ch    = static_mask.expand_as(ego_flow)       # [S-1, 2, H, W]

        n_static = mask_2ch.sum()
        if n_static == 0:
            return torch.zeros(1, device=depth.device).squeeze()

        return F.smooth_l1_loss(ego_flow[mask_2ch], self.flow_fwd[mask_2ch])

    def _depth_reg_loss(self, depth: torch.Tensor) -> torch.Tensor:
        """Scale-invariant depth regularization toward VGGT initialization."""
        return depth_regularization_si_weighted(
            depth.unsqueeze(1),           # [S, 1, H, W]
            self.init_depth.unsqueeze(1), # [S, 1, H, W]
        )

    # ------------------------------------------------------------------
    def _global_smooth_loss(self, refined_pts: torch.Tensor) -> torch.Tensor:
        """Global 3D track smoothness loss for ALL pixels (mon, MonST3R-style).

        Constant-velocity prior applied without any dynamic mask:

            L_smooth = Σ_{t=1}^{S-2} Σ_i ‖(X^{t+1}-X^t) - (X^t-X^{t-1})‖_2^2
        """
        if self.S < 3:
            return torch.zeros(1, device=refined_pts.device).squeeze()

        vel_fwd = refined_pts[2:]   - refined_pts[1:-1]   # [S-2, H, W, 3]
        vel_bwd = refined_pts[1:-1] - refined_pts[:-2]    # [S-2, H, W, 3]
        accel   = (vel_fwd - vel_bwd).pow(2).sum(dim=-1)  # [S-2, H, W]
        return accel.mean()

    # ------------------------------------------------------------------
    # dyn losses
    # ------------------------------------------------------------------

    def _static_rigid_loss(self, refined_pts: torch.Tensor) -> torch.Tensor:
        """Static rigid alignment loss (dyn).

        For static pixels in adjacent pairs, world points at t and t+1
        should be equal (same physical point in world frame).

            L_static = Σ_{(t,t+1)} Σ_i (1-M_i^t)·C_i^t·‖X_i^t - X_i^{t+1}‖_1
        """
        if self.S < 2:
            return torch.zeros(1, device=refined_pts.device).squeeze()

        pts_t  = refined_pts[:-1]                           # [S-1, H, W, 3]
        pts_t1 = refined_pts[1:]                            # [S-1, H, W, 3]

        diff     = (pts_t - pts_t1).abs().sum(dim=-1)       # [S-1, H, W]  L1
        static_w = 1.0 - self.dynamic_mask.squeeze(1).float()  # [S-1, H, W]
        conf_t   = self.anchor_conf[:-1]                    # [S-1, H, W]

        weighted = static_w * conf_t * diff
        n = static_w.sum().clamp(min=1.0)
        return weighted.sum() / n

    def _dyn_pointmap_loss(self, refined_pts: torch.Tensor) -> torch.Tensor:
        """Dynamic pointmap regression loss (dyn, MonST3R-style).

        For dynamic pixels, regress refined world points toward VGGT
        world_points (used as pseudo-GT), weighted by confidence.

            L_dyn = Σ_t Σ_i M_i^t · C_i^t · ‖X_i^t - X̄_i^t‖_1

        Note: confidence maximization term (−α log C) is omitted because
        anchor_conf is a fixed VGGT buffer, not a learnable parameter.
        """
        S, H, W = self.S, self.H, self.W
        device  = refined_pts.device

        # Per-frame dynamic mask via OR of adjacent pair masks
        pm         = self.dynamic_mask.squeeze(1)              # [S-1, H, W]
        frame_dyn  = torch.zeros(S, H, W, dtype=torch.bool, device=device)
        frame_dyn[:-1] |= pm
        frame_dyn[1:]  |= pm

        dyn_w    = frame_dyn.float()                           # [S, H, W]
        diff     = (refined_pts - self.anchor_pts).abs().sum(dim=-1)  # [S, H, W]
        weighted = dyn_w * self.anchor_conf * diff
        n        = dyn_w.sum().clamp(min=1.0)
        return weighted.sum() / n

    def _track_smooth_loss(self, refined_pts: torch.Tensor) -> torch.Tensor:
        """3D track smoothness loss for dynamic points (dyn).

        Constant-velocity prior: penalise acceleration of dynamic-point trajectories.

            L_smooth = Σ_{t=1}^{S-2} Σ_i M_i^t · ‖(X^{t+1}-X^t) - (X^t-X^{t-1})‖_2^2
        """
        if self.S < 3:
            return torch.zeros(1, device=refined_pts.device).squeeze()

        vel_fwd = refined_pts[2:]   - refined_pts[1:-1]   # [S-2, H, W, 3]
        vel_bwd = refined_pts[1:-1] - refined_pts[:-2]    # [S-2, H, W, 3]
        accel   = (vel_fwd - vel_bwd).pow(2).sum(dim=-1)  # [S-2, H, W]

        # Middle-frame dynamic mask: pair (t-1,t) OR pair (t,t+1)
        pm      = self.dynamic_mask.squeeze(1)            # [S-1, H, W]
        dyn_mid = pm[:-1] | pm[1:]                        # [S-2, H, W]

        weighted = dyn_mid.float() * accel
        n        = dyn_mid.float().sum().clamp(min=1.0)
        return weighted.sum() / n

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, epoch: int = 0):
        """Compute total loss.

        Dispatches to _forward_mon or _forward_dyn based on self.loss_version.

        Returns:
            (total_loss, secondary_loss) — both scalar tensors
        """
        if self.loss_version == "dyn":
            return self._forward_dyn()
        return self._forward_mon()

    def _forward_mon(self):
        """mon: anchor + ego-flow + depth-reg + global track smooth."""
        depth = self.get_refined_depth()
        R, T  = self.get_refined_RT()

        refined_pts = self._depth_to_world(depth, R, T)

        anchor_loss = self._anchor_loss(refined_pts)
        flow_loss   = self._flow_loss(depth, R, T)
        depth_loss  = self._depth_reg_loss(depth)
        smooth_loss = self._global_smooth_loss(refined_pts)

        total_loss = (
            self.anchor_weight       * anchor_loss
            + self.flow_weight       * flow_loss
            + self.depth_reg_weight  * depth_loss
            + self.mon_smooth_weight * smooth_loss
        )
        return total_loss, flow_loss.detach()

    def _forward_dyn(self):
        """dyn: static-rigid + λ·dyn_pointmap + γ·track_smooth."""
        depth = self.get_refined_depth()
        R, T  = self.get_refined_RT()

        refined_pts = self._depth_to_world(depth, R, T)

        static_loss  = self._static_rigid_loss(refined_pts)
        dyn_loss     = self._dyn_pointmap_loss(refined_pts)
        smooth_loss  = self._track_smooth_loss(refined_pts)

        total_loss = (
            static_loss
            + self.dyn_pointmap_weight * dyn_loss
            + self.track_smooth_weight * smooth_loss
        )
        return total_loss, dyn_loss.detach()

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
