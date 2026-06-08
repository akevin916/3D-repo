"""evaluators/metrics.py — shared metric functions.

Depth metrics (per-pixel, after masking and optional scale alignment):
    AbsRel, SqRel, RMSE, RMSElog, δ1 / δ2 / δ3

Point-cloud metrics (Chamfer-based):
    accuracy_mm, completeness_mm, chamfer_mm, threshold recall %
"""

import numpy as np


# ---------------------------------------------------------------------------
# Depth metrics (standard monocular depth protocol)
# ---------------------------------------------------------------------------

def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Compute standard depth evaluation metrics.

    Args:
        pred: predicted depth [H, W] or (N,), positive floats, same unit as gt
        gt  : ground-truth depth [H, W] or (N,), positive floats

    Returns:
        dict with keys: abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3
    """
    pred = pred.ravel().astype(np.float64)
    gt   = gt.ravel().astype(np.float64)

    assert pred.shape == gt.shape, "pred and gt must have the same shape"
    assert len(pred) > 0, "no valid pixels"

    thresh = np.maximum(pred / gt, gt / pred)       # (N,)

    abs_rel  = np.mean(np.abs(pred - gt) / gt)
    sq_rel   = np.mean(((pred - gt) ** 2) / gt)
    rmse     = np.sqrt(np.mean((pred - gt) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(pred) - np.log(gt)) ** 2))
    d1 = float(np.mean(thresh < 1.25))
    d2 = float(np.mean(thresh < 1.25 ** 2))
    d3 = float(np.mean(thresh < 1.25 ** 3))

    return {
        "abs_rel":  float(abs_rel),
        "sq_rel":   float(sq_rel),
        "rmse":     float(rmse),
        "rmse_log": float(rmse_log),
        "d1":       d1,
        "d2":       d2,
        "d3":       d3,
    }


def median_scale_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Scale pred so that median(pred_valid) == median(gt_valid).

    Args:
        pred: predicted depth array (any shape), positive
        gt  : ground-truth depth array (same shape), positive; zero = invalid

    Returns:
        Scaled pred (same shape).
    """
    valid = gt > 0
    if valid.sum() == 0:
        return pred
    scale = np.median(gt[valid]) / (np.median(pred[valid]) + 1e-8)
    return pred * scale


def scale_only_fit(pred: np.ndarray,
                   gt: np.ndarray,
                   n_iters: int = 10,
                   eps: float = 1e-8) -> float:
    """Fit one global scale s for pred->gt without shift.

    Uses a robust iterative reweighted update inspired by MonST3R's
    ``align_with_scale`` branch.

    Args:
        pred: predicted depth values (any shape)
        gt  : ground-truth depth values (same shape)
        n_iters: number of IRLS updates
        eps: numerical stabilizer

    Returns:
        scalar s such that pred_aligned = s * pred
    """
    p = pred.ravel().astype(np.float64)
    g = gt.ravel().astype(np.float64)
    valid = (g > 0) & np.isfinite(g) & np.isfinite(p)
    if valid.sum() == 0:
        return 1.0

    p = p[valid]
    g = g[valid]
    s = float(np.mean(g) / (np.mean(p) + eps))

    for _ in range(max(1, n_iters)):
        residual = s * p - g
        w = 1.0 / (np.abs(residual) + eps)
        num = np.sum(w * p * g)
        den = np.sum(w * p * p) + eps
        s = float(num / den)

    return max(s, 1e-6)


def scale_shift_align(pred: np.ndarray, gt: np.ndarray,
                      lr: float = 1e-4, max_iters: int = 1000,
                      tol: float = 1e-6) -> np.ndarray:
    """Scale + shift alignment matching MonST3R's align_with_lad2 protocol.

    Finds s, t minimising L1( s*pred + t - gt ) via Adam, applied to all
    valid pixels at once (same as MonST3R's per-sequence alignment).

    Args:
        pred: predicted depth array (any shape), positive
        gt  : ground-truth depth array (same shape), positive; zero = invalid

    Returns:
        Aligned pred (same shape): s * pred + t
    """
    import torch

    valid = (gt > 0).ravel()
    p = torch.tensor(pred.ravel()[valid], dtype=torch.float32)
    g = torch.tensor(gt.ravel()[valid],   dtype=torch.float32)

    s_init = float(torch.median(g) / (torch.median(p) + 1e-8))
    s = torch.tensor([s_init], requires_grad=True, dtype=torch.float32)
    t = torch.tensor([0.0],    requires_grad=True, dtype=torch.float32)

    opt = torch.optim.Adam([s, t], lr=lr)
    prev_loss = None

    for _ in range(max_iters):
        opt.zero_grad()
        loss = torch.abs(s * p + t - g).sum()
        loss.backward()
        opt.step()
        cur = loss.item()
        if prev_loss is not None and abs(prev_loss - cur) < tol:
            break
        prev_loss = cur

    s_val = s.detach().item()
    t_val = t.detach().item()
    return (pred * s_val + t_val).astype(np.float32)


# ---------------------------------------------------------------------------
# Point-cloud Chamfer metrics
# ---------------------------------------------------------------------------

def chamfer_metrics(pred_pts: np.ndarray,
                    gt_pts: np.ndarray,
                    thresholds: tuple = (1.0, 2.0, 5.0, 10.0)) -> dict:
    """Compute Accuracy, Completeness, Chamfer and threshold recall.

    Args:
        pred_pts: (N, 3) predicted point cloud
        gt_pts  : (M, 3) ground-truth point cloud
        thresholds: distance thresholds in the same unit as the point clouds

    Returns:
        dict with accuracy_mm, completeness_mm, chamfer_mm,
        acc_under_{t}mm_%, comp_under_{t}mm_%
    """
    from scipy.spatial import cKDTree

    pred_tree = cKDTree(pred_pts)
    gt_tree   = cKDTree(gt_pts)

    acc_dists,  _ = gt_tree.query(pred_pts)    # pred → GT  (accuracy)
    comp_dists, _ = pred_tree.query(gt_pts)    # GT → pred  (completeness)

    results = {
        "accuracy_mm":     float(np.mean(acc_dists)),
        "completeness_mm": float(np.mean(comp_dists)),
        "chamfer_mm":      float((np.mean(acc_dists) + np.mean(comp_dists)) / 2.0),
    }
    for t in thresholds:
        results[f"acc_under_{t}mm_%"]  = float(np.mean(acc_dists  < t) * 100)
        results[f"comp_under_{t}mm_%"] = float(np.mean(comp_dists < t) * 100)

    return results


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_metrics(results: dict, title: str = "Results"):
    w = max(len(k) for k in results) + 2
    print(f"\n{'─' * (w + 14)}")
    print(f"  {title}")
    print(f"{'─' * (w + 14)}")
    for k, v in results.items():
        print(f"  {k:<{w}} {v:.4f}")
    print(f"{'─' * (w + 14)}\n")
