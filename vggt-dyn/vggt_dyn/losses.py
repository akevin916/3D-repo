"""
losses.py — Loss strategies for VGGTDynOptimizer.

Each class is a self-contained nn.Module whose forward(opt, epoch) receives the
optimizer instance to access buffers (anchor_pts, dynamic_mask, …) and geometry
helpers (_depth_to_world, K, …).

    MonLoss — anchor + ego-flow + depth-reg + pose-smooth
    DynLoss — static-rigid + dyn-pointmap + track-smooth
"""

import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from third_party.goem_opt import DepthBasedWarping, depth_regularization_si_weighted


# ---------------------------------------------------------------------------
# MonLoss
# ---------------------------------------------------------------------------

class MonLoss(nn.Module):
    """MonST3R-style loss for dynamic scenes.

    Args:
        anchor_weight       : weight for anchor loss
        flow_weight         : weight for ego-flow loss
        depth_reg_weight    : weight for depth regularization
        smooth_weight       : weight for pose smoothness loss
        scale_flow_loss     : use scale-normalised flow loss instead of plain
        translation_weight  : translation term weight inside pose smooth loss
        niter               : total optimisation iterations (used for warmup)
        flow_loss_start_frac: skip flow loss for the first this fraction of iters
        flow_loss_thre      : global safety switch — zero flow loss above this value
        pxl_thre            : per-pixel loss threshold to drop outliers (0 = off)
    """

    def __init__(
        self,
        anchor_weight: float        = 1.0,
        flow_weight: float          = 1.0,
        depth_reg_weight: float     = 0.1,
        smooth_weight: float        = 0.1,
        scale_flow_loss: bool       = False,
        translation_weight: float   = 0.1,
        niter: int                  = 300,
        flow_loss_start_frac: float = 0.15,
        flow_loss_thre: float       = 50.0,
        pxl_thre: float             = 50.0,
    ):
        super().__init__()
        self.anchor_weight        = anchor_weight
        self.flow_weight          = flow_weight
        self.depth_reg_weight     = depth_reg_weight
        self.smooth_weight        = smooth_weight
        self.scale_flow_loss      = scale_flow_loss
        self.translation_weight   = translation_weight
        self.niter                = niter
        self.flow_loss_start_frac = flow_loss_start_frac
        self.flow_loss_thre       = flow_loss_thre
        self.pxl_thre             = pxl_thre

        # DepthBasedWarping is only used by flow losses, so it lives here.
        self._depth_warping = DepthBasedWarping()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, opt, epoch: int = 0):
        """Compute mon total loss.

        Args:
            opt  : VGGTDynOptimizer instance (buffers + geometry helpers)
            epoch: current iteration index (for warmup)

        Returns:
            (total_loss, flow_loss) — both scalar tensors
        """
        depth = opt.get_refined_depth()
        R, T  = opt.get_refined_RT()
        reproj_pts = opt._depth_to_world(depth, R, T)

        anchor_loss = self._anchor_loss(opt, reproj_pts)

        # warmup: skip flow loss for first flow_loss_start_frac of iterations
        if self.flow_loss_start_frac > 0 and epoch < int(self.niter * self.flow_loss_start_frac):
            flow_loss = torch.zeros(1, device=depth.device).squeeze()
        else:
            flow_fn   = self._flow_loss_scale_normalized if self.scale_flow_loss else self._flow_loss
            flow_loss = flow_fn(opt, depth, R, T)
            # global safety switch: disable flow loss when scene is highly dynamic
            if self.flow_loss_thre > 0 and float(flow_loss) > self.flow_loss_thre:
                flow_loss = torch.zeros(1, device=depth.device).squeeze()

        depth_loss  = self._depth_reg_loss(opt, depth)
        smooth_loss = self._pose_smooth_loss(opt, R, T)

        total = (
            self.anchor_weight      * anchor_loss
            + self.flow_weight      * flow_loss
            + self.depth_reg_weight * depth_loss
            + self.smooth_weight    * smooth_loss
        )
        return total, flow_loss.detach()

    # ------------------------------------------------------------------
    # individual losses
    # ------------------------------------------------------------------

    def _anchor_loss(self, opt, reproj_pts: torch.Tensor) -> torch.Tensor:
        """Weighted L2 between reprojected world pts and VGGT anchor.

        Loss = mean(anchor_conf * ||reproj_pts - anchor_pts||^2)
        """
        diff    = reproj_pts - opt.anchor_pts       # [S, H, W, 3]
        sq_dist = (diff ** 2).sum(dim=-1)           # [S, H, W]
        return (opt.anchor_conf * sq_dist).mean()

    def _flow_loss(
        self,
        opt,
        depth: torch.Tensor,   # [S, H, W]
        R: torch.Tensor,       # [S, 3, 3]
        T: torch.Tensor,       # [S, 3]
    ) -> torch.Tensor:
        """Ego-flow consistency loss on static pixels only."""
        if opt.S < 2:
            return torch.zeros(1, device=depth.device).squeeze()

        K     = opt.K[:-1]
        K_inv = torch.linalg.inv(K)

        R_c2w = R.transpose(-1, -2)
        t_c2w = -(R_c2w @ T.unsqueeze(-1))

        ego_hom, _ = self._depth_warping(
            R_c2w[:-1], t_c2w[:-1], R_c2w[1:], t_c2w[1:],
            depth[:-1].unsqueeze(1), K, K_inv, use_depth=True,
        )
        ego_flow = ego_hom[:, :2]   # [S-1, 2, H, W]

        static_mask = opt.valid_fwd & ~opt.dynamic_mask   # [S-1, 1, H, W]
        mask_2ch    = static_mask.expand_as(ego_flow)

        n_static = mask_2ch.sum()
        if n_static == 0:
            return torch.zeros(1, device=depth.device).squeeze()

        loss_per_px = F.smooth_l1_loss(
            ego_flow[mask_2ch], opt.flow_fwd[mask_2ch], reduction='none'
        )
        if self.pxl_thre > 0:
            valid = loss_per_px < self.pxl_thre
            if valid.sum() == 0:
                return torch.zeros(1, device=depth.device).squeeze()
            return loss_per_px[valid].mean()
        return loss_per_px.mean()

    def _flow_loss_scale_normalized(
        self,
        opt,
        depth: torch.Tensor,
        R: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        """Scale-normalised ego-flow loss.

        Applies a per-pair stop-gradient scale correction so that ego_flow and
        RAFT_flow share the same global magnitude per pair.  Removes the penalty
        for depth/translation scale bias while keeping the spatial-pattern
        gradient that corrects relative pose.

        scale_i = mean(|RAFT_i|) / mean(|ego_i|)  [stop-gradient]
        loss    = smooth_l1(scale_i * ego_flow_i, RAFT_flow_i)  per static pixel
        """
        if opt.S < 2:
            return torch.zeros(1, device=depth.device).squeeze()

        K     = opt.K[:-1]
        K_inv = torch.linalg.inv(K)
        R_c2w = R.transpose(-1, -2)
        t_c2w = -(R_c2w @ T.unsqueeze(-1))

        ego_hom, _ = self._depth_warping(
            R_c2w[:-1], t_c2w[:-1], R_c2w[1:], t_c2w[1:],
            depth[:-1].unsqueeze(1), K, K_inv, use_depth=True,
        )
        ego_flow = ego_hom[:, :2]   # [S-1, 2, H, W]

        static_mask = opt.valid_fwd & ~opt.dynamic_mask   # [S-1, 1, H, W]

        pair_losses = []
        for i in range(opt.S - 1):
            mask_i = static_mask[i].expand(2, opt.H, opt.W)   # [2, H, W]
            if mask_i.sum() == 0:
                continue
            ego_i  = ego_flow[i][mask_i]        # [N*2]
            raft_i = opt.flow_fwd[i][mask_i]    # [N*2]

            with torch.no_grad():
                ego_mag  = ego_i.abs().mean().clamp(min=1e-6)
                raft_mag = raft_i.abs().mean().clamp(min=1e-6)
                scale    = (raft_mag / ego_mag).clamp(0.05, 20.0)

            loss_i = F.smooth_l1_loss(scale * ego_i, raft_i, reduction='none')
            if self.pxl_thre > 0:
                valid = loss_i < self.pxl_thre
                if valid.sum() == 0:
                    continue
                pair_losses.append(loss_i[valid].mean())
            else:
                pair_losses.append(loss_i.mean())

        if not pair_losses:
            return torch.zeros(1, device=depth.device).squeeze()
        return torch.stack(pair_losses).mean()

    def _depth_reg_loss(self, opt, depth: torch.Tensor) -> torch.Tensor:
        """Scale-invariant depth regularization toward VGGT init depth.

        Dynamic pixels receive 2x penalty to prevent unconstrained depth drift
        in regions the optimizer cannot correct via static-scene cues.
        """
        S, H, W = opt.S, opt.H, opt.W
        frame_dyn = torch.zeros(S, 1, H, W, device=depth.device)
        if S > 1:
            pm = opt.dynamic_mask.float()              # [S-1, 1, H, W]
            frame_dyn[:-1] = torch.maximum(frame_dyn[:-1], pm)
            frame_dyn[1:]  = torch.maximum(frame_dyn[1:],  pm)
        return depth_regularization_si_weighted(
            depth.unsqueeze(1),
            opt.init_depth.unsqueeze(1),
            pixel_wise_weight=frame_dyn,
        )

    def _pose_smooth_loss(self, opt, R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Small-change prior on adjacent camera poses.

        L_rot   = ||R[t+1] @ R[t]^T - I||_F^2
        L_trans = ||C[t+1] - C[t]||^2,  C[t] = -R[t]^T @ T[t]
        """
        if opt.S < 2:
            return torch.zeros(1, device=R.device).squeeze()

        R_rel = R[1:] @ R[:-1].transpose(-1, -2)            # [S-1, 3, 3]
        I     = torch.eye(3, device=R.device).unsqueeze(0)  # [1, 3, 3]
        L_rot = (R_rel - I).pow(2).sum(dim=(-2, -1))        # [S-1]

        C       = -(R.transpose(-1, -2) @ T.unsqueeze(-1)).squeeze(-1)  # [S, 3]
        L_trans = (C[1:] - C[:-1]).pow(2).sum(dim=-1)                   # [S-1]

        return (L_rot + self.translation_weight * L_trans).mean()


# ---------------------------------------------------------------------------
# DynLoss
# ---------------------------------------------------------------------------

class DynLoss(nn.Module):
    """Dynamic-scene loss using per-pixel rigidity/motion decomposition.

    Args:
        dyn_pointmap_weight : λ — weight for dynamic pointmap regression loss
        track_smooth_weight : γ — weight for 3D track smoothness on dynamic pixels
    """

    def __init__(
        self,
        dyn_pointmap_weight: float = 1.0,
        track_smooth_weight: float = 0.1,
    ):
        super().__init__()
        self.dyn_pointmap_weight = dyn_pointmap_weight
        self.track_smooth_weight = track_smooth_weight

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, opt, epoch: int = 0):
        """Compute dyn total loss.

        Args:
            opt  : VGGTDynOptimizer instance
            epoch: current iteration index (unused, kept for interface parity)

        Returns:
            (total_loss, dyn_loss) — both scalar tensors
        """
        depth = opt.get_refined_depth()
        R, T  = opt.get_refined_RT()
        reproj_pts = opt._depth_to_world(depth, R, T)

        static_loss = self._static_rigid_loss(opt, reproj_pts)
        dyn_loss    = self._dyn_pointmap_loss(opt, reproj_pts)
        smooth_loss = self._track_smooth_loss(opt, reproj_pts)

        total = (
            static_loss
            + self.dyn_pointmap_weight * dyn_loss
            + self.track_smooth_weight * smooth_loss
        )
        return total, dyn_loss.detach()

    # ------------------------------------------------------------------
    # individual losses
    # ------------------------------------------------------------------

    def _static_rigid_loss(self, opt, reproj_pts: torch.Tensor) -> torch.Tensor:
        """Static rigidity: world points at (t, t+1) should be equal for static pixels.

        L_static = Σ_{(t,t+1)} Σ_i (1-M_i^t)·C_i^t·‖X_i^t - X_i^{t+1}‖_1
        """
        if opt.S < 2:
            return torch.zeros(1, device=reproj_pts.device).squeeze()

        pts_t  = reproj_pts[:-1]                                    # [S-1, H, W, 3]
        pts_t1 = reproj_pts[1:]                                     # [S-1, H, W, 3]

        diff     = (pts_t - pts_t1).abs().sum(dim=-1)               # [S-1, H, W]
        static_w = 1.0 - opt.dynamic_mask.squeeze(1).float()       # [S-1, H, W]
        conf_t   = opt.anchor_conf[:-1]                             # [S-1, H, W]

        weighted = static_w * conf_t * diff
        n        = static_w.sum().clamp(min=1.0)
        return weighted.sum() / n

    def _dyn_pointmap_loss(self, opt, reproj_pts: torch.Tensor) -> torch.Tensor:
        """Regress dynamic-pixel world points toward VGGT pseudo-GT.

        L_dyn = Σ_t Σ_i M_i^t · C_i^t · ‖X_i^t - X̄_i^t‖_1
        """
        S, H, W = opt.S, opt.H, opt.W
        device  = reproj_pts.device

        pm        = opt.dynamic_mask.squeeze(1)              # [S-1, H, W]
        frame_dyn = torch.zeros(S, H, W, dtype=torch.bool, device=device)
        frame_dyn[:-1] |= pm
        frame_dyn[1:]  |= pm

        dyn_w    = frame_dyn.float()
        diff     = (reproj_pts - opt.anchor_pts).abs().sum(dim=-1)  # [S, H, W]
        weighted = dyn_w * opt.anchor_conf * diff
        n        = dyn_w.sum().clamp(min=1.0)
        return weighted.sum() / n

    def _track_smooth_loss(self, opt, reproj_pts: torch.Tensor) -> torch.Tensor:
        """Constant-velocity prior on dynamic-pixel 3D trajectories.

        L_smooth = Σ_{t=1}^{S-2} Σ_i M_i^t · ‖(X^{t+1}-X^t) - (X^t-X^{t-1})‖_2^2
        """
        if opt.S < 3:
            return torch.zeros(1, device=reproj_pts.device).squeeze()

        vel_fwd = reproj_pts[2:]   - reproj_pts[1:-1]  # [S-2, H, W, 3]
        vel_bwd = reproj_pts[1:-1] - reproj_pts[:-2]   # [S-2, H, W, 3]
        accel   = (vel_fwd - vel_bwd).pow(2).sum(dim=-1)  # [S-2, H, W]

        pm      = opt.dynamic_mask.squeeze(1)           # [S-1, H, W]
        dyn_mid = pm[:-1] | pm[1:]                      # [S-2, H, W]

        weighted = dyn_mid.float() * accel
        n        = dyn_mid.float().sum().clamp(min=1.0)
        return weighted.sum() / n
