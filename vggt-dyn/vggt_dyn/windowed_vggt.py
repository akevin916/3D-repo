"""windowed_vggt.py — Multi-window VGGT forward + Pose Graph Optimization.

Runs VGGT on overlapping temporal windows, aligns them with a pose graph
optimizer, and merges into a single globally consistent anchor that can
replace the single-pass VGGT output fed into VGGTInitializer / TTO.

Public API
----------
    from vggt_dyn.windowed_vggt import run_windowed_vggt

    merged_state = run_windowed_vggt(
        model, image_paths, device, preprocess="center_crop",
        window_size=20, stride=5,
    )
    # merged_state is a dict with keys: R, T, K, depth, depth_conf, anchor_pts, anchor_conf
    # Drop it directly into VGGTInitializer._state or pass to from_state().
"""

import os
import sys
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as _Rot

_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_MODULE_DIR)
_VGGT_REPO   = os.path.join(_PROJECT_DIR, "vggt")
for _p in (_PROJECT_DIR, _VGGT_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ===========================================================================
# Window generation
# ===========================================================================

def _generate_windows(S: int, window_size: int, stride: int):
    """Return list of (start, end) covering all S frames.

    The last window is extended to reach frame S-1 if needed so no frames
    are left out at the tail.
    """
    windows = []
    start = 0
    while start < S:
        end = min(start + window_size, S)
        windows.append((start, end))
        if end == S:
            break
        start += stride
    return windows


def _center_weight(local_idx: int, window_size: int) -> float:
    """Triangular weight: center frame → 1.0, edge frames → 0.1 (never 0).

    The 0.1 floor ensures frames that appear only in one window always get
    a non-zero weight, preventing a zero K matrix and singular inversion.
    """
    if window_size <= 1:
        return 1.0
    half = (window_size - 1) / 2.0
    return max(0.1, 1.0 - abs(local_idx - half) / half)


# ===========================================================================
# Rotation helpers
# ===========================================================================

def _rvec_to_mat(rvec: np.ndarray) -> np.ndarray:
    return _Rot.from_rotvec(rvec).as_matrix()


def _mat_to_rvec(R: np.ndarray) -> np.ndarray:
    return _Rot.from_matrix(R).as_rotvec()


def _svd_orthonorm(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3) via SVD."""
    U, _, Vt = np.linalg.svd(R)
    Ro = U @ Vt
    if np.linalg.det(Ro) < 0:
        U[:, -1] *= -1
        Ro = U @ Vt
    return Ro


# ===========================================================================
# Pose graph optimization
# ===========================================================================

def _solve_pose_graph(window_outputs: list) -> list:
    """Find SE3 alignment (R_align, t_align) per window.

    Window 0 is the reference (identity). Minimizes the discrepancy of
    camera positions and orientations for frames shared across windows.

    Returns
    -------
    list of (R_align [3,3], t_align [3]) — one per window.
    """
    N = len(window_outputs)
    if N == 1:
        return [(np.eye(3), np.zeros(3))]

    # Pre-compute camera centers c = -R^T @ T  and mean confidence per frame
    cam_centers = []
    mean_confs  = []
    for wo in window_outputs:
        R_np   = wo["R"].cpu().numpy()    # [W, 3, 3]
        T_np   = wo["T"].cpu().numpy()    # [W, 3]
        conf_np = wo["conf"].cpu().numpy() # [W, H, Wi]
        centers = np.stack([-R_np[f].T @ T_np[f] for f in range(len(R_np))])
        cam_centers.append(centers)                        # [W, 3]
        mean_confs.append(conf_np.mean(axis=(1, 2)))       # [W]

    # Build edge list: all pairs of windows that share at least one frame
    edges = []
    for k in range(N):
        sk, ek = window_outputs[k]["start"], window_outputs[k]["end"]
        for l in range(k + 1, N):
            sl, el = window_outputs[l]["start"], window_outputs[l]["end"]
            shared = sorted(set(range(sk, ek)) & set(range(sl, el)))
            if shared:
                edges.append({
                    "k": k, "l": l,
                    "local_k": [f - sk for f in shared],
                    "local_l": [f - sl for f in shared],
                })

    if not edges:
        print("[windowed-vggt] no overlapping windows; alignments set to identity")
        return [(np.eye(3), np.zeros(3))] * N

    # Optimization: variables x = [rvec_1, t_1, rvec_2, t_2, ...] for k=1..N-1
    # (window 0 is fixed)
    def unpack(x):
        R_list = [np.eye(3)]
        t_list = [np.zeros(3)]
        for i in range(N - 1):
            R_list.append(_rvec_to_mat(x[i * 6     : i * 6 + 3]))
            t_list.append(             x[i * 6 + 3 : i * 6 + 6])
        return R_list, t_list

    def objective(x):
        R_list, t_list = unpack(x)
        loss = 0.0
        for edge in edges:
            k, l = edge["k"], edge["l"]
            Ra_k, ta_k = R_list[k], t_list[k]
            Ra_l, ta_l = R_list[l], t_list[l]
            R_k = window_outputs[k]["R"].cpu().numpy()
            T_k = window_outputs[k]["T"].cpu().numpy()
            R_l = window_outputs[l]["R"].cpu().numpy()
            T_l = window_outputs[l]["T"].cpu().numpy()
            Wk  = window_outputs[k]["end"] - window_outputs[k]["start"]
            Wl  = window_outputs[l]["end"] - window_outputs[l]["start"]

            for jk, jl in zip(edge["local_k"], edge["local_l"]):
                # --- camera center in global coords ---
                # global_center = R_align @ local_center + t_align
                cg_k = Ra_k @ cam_centers[k][jk] + ta_k
                cg_l = Ra_l @ cam_centers[l][jl] + ta_l
                pos_err = np.sum((cg_k - cg_l) ** 2)

                # --- camera rotation in global coords ---
                # R_global = R_cam_local @ R_align^T
                Rg_k = R_k[jk] @ Ra_k.T
                Rg_l = R_l[jl] @ Ra_l.T
                rot_err = np.sum((Rg_k - Rg_l) ** 2)

                w = (float(mean_confs[k][jk]) * float(mean_confs[l][jl]) *
                     _center_weight(jk, Wk) * _center_weight(jl, Wl))
                loss += w * (pos_err + rot_err)
        return loss

    x0 = np.zeros(6 * (N - 1), dtype=np.float64)
    res = minimize(objective, x0, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7})
    print(f"[windowed-vggt] pose graph: loss={res.fun:.4e}  success={res.success}")

    R_list, t_list = unpack(res.x)
    return list(zip(R_list, t_list))


# ===========================================================================
# Output merging
# ===========================================================================

def _merge_outputs(window_outputs: list, alignments: list,
                   S_total: int, device) -> dict:
    """Weighted merge of aligned window outputs into a single state dict.

    Each frame accumulates weighted contributions from every window that
    contains it. Weights = mean_conf × center_weight. Rotation matrices
    are re-orthonormalised via SVD after averaging.
    """
    # Lazy init accumulators (shapes not known until first window processed)
    R_accum       = None
    T_accum       = None
    K_accum       = None
    depth_accum   = None
    pts3d_accum   = None
    conf_accum    = None
    dconf_accum   = None
    w_accum       = np.zeros(S_total)

    for k, wo in enumerate(window_outputs):
        Ra, ta = alignments[k]             # [3,3], [3]
        R_np   = wo["R"].cpu().numpy()     # [Wk, 3, 3]
        T_np   = wo["T"].cpu().numpy()     # [Wk, 3]
        K_np   = wo["K"].cpu().numpy()     # [Wk, 3, 3]
        d_np   = wo["depth"].cpu().numpy() # [Wk, H, Wi]
        p_np   = wo["pts3d"].cpu().numpy() # [Wk, H, Wi, 3]
        c_np   = wo["conf"].cpu().numpy()  # [Wk, H, Wi]
        dc_np  = wo["depth_conf"].cpu().numpy()  # [Wk, H, Wi]

        if R_accum is None:
            H_img, Wi = d_np.shape[1], d_np.shape[2]
            R_accum     = np.zeros((S_total, 3, 3))
            T_accum     = np.zeros((S_total, 3))
            K_accum     = np.zeros((S_total, 3, 3))
            depth_accum = np.zeros((S_total, H_img, Wi))
            pts3d_accum = np.zeros((S_total, H_img, Wi, 3))
            conf_accum  = np.zeros((S_total, H_img, Wi))
            dconf_accum = np.zeros((S_total, H_img, Wi))

        start, end = wo["start"], wo["end"]
        Wk = end - start

        for j in range(Wk):
            gi = start + j
            cw = _center_weight(j, Wk)
            w  = float(cw * c_np[j].mean())

            # Camera pose in global coords:
            #   R_global = R_local @ R_align^T
            #   T_global = T_local - R_local @ R_align^T @ t_align
            Rg = R_np[j] @ Ra.T
            Tg = T_np[j] - R_np[j] @ Ra.T @ ta

            # World points in global coords:
            #   p_global = R_align @ p_local + t_align
            #   vectorised: pts3d_g[h,w] = pts3d[h,w] @ R_align^T + t_align
            pg = p_np[j] @ Ra.T + ta           # [H, Wi, 3]

            R_accum[gi]     += w * Rg
            T_accum[gi]     += w * Tg
            K_accum[gi]     += w * K_np[j]
            depth_accum[gi] += w * d_np[j]
            pts3d_accum[gi] += w * pg
            conf_accum[gi]  += w * c_np[j]
            dconf_accum[gi] += w * dc_np[j]
            w_accum[gi]     += w

    # Normalize
    ws = np.maximum(w_accum, 1e-8)
    R_mean     = R_accum     / ws[:, None, None]
    T_mean     = T_accum     / ws[:, None]
    K_mean     = K_accum     / ws[:, None, None]
    depth_mean = depth_accum / ws[:, None, None]
    pts3d_mean = pts3d_accum / ws[:, None, None, None]
    conf_mean  = conf_accum  / ws[:, None, None]
    dconf_mean = dconf_accum / ws[:, None, None]

    # Re-orthonormalise rotation matrices
    for i in range(S_total):
        R_mean[i] = _svd_orthonorm(R_mean[i])

    def _t(arr):
        return torch.from_numpy(arr.astype(np.float32)).to(device)

    return {
        "R":           _t(R_mean),
        "T":           _t(T_mean),
        "K":           _t(K_mean),
        "depth":       _t(depth_mean),
        "depth_conf":  _t(dconf_mean),
        "anchor_pts":  _t(pts3d_mean),
        "anchor_conf": _t(conf_mean),
    }


# ===========================================================================
# Public entry point
# ===========================================================================

def run_windowed_vggt(model, image_paths: list, device,
                      preprocess: str = "center_crop",
                      window_size: int = 20,
                      stride: int = 5) -> dict:
    """Multi-window VGGT forward + pose graph optimization.

    Parameters
    ----------
    model       : loaded VGGT model (eval mode, on device)
    image_paths : ordered list of absolute image paths (length S)
    device      : torch.device
    preprocess  : same as run_vggt's preprocess argument
    window_size : number of frames per window (W)
    stride      : step between window starts

    Returns
    -------
    dict with keys matching VGGTInitializer._state:
        R [S,3,3], T [S,3], K [S,3,3],
        depth [S,H,W], depth_conf [S,H,W],
        anchor_pts [S,H,W,3], anchor_conf [S,H,W]
    """
    from vggt_dyn.pipeline import run_vggt
    from vggt_dyn.initializer import VGGTInitializer
    from vggt_dyn.utils.pose_utils import decode_pose_enc

    S = len(image_paths)

    # Degenerate case: window covers the full sequence → single forward
    if window_size >= S:
        print(f"[windowed-vggt] window_size={window_size} >= S={S}, "
              f"falling back to single forward pass")
        with torch.no_grad():
            vggt_out = run_vggt(model, image_paths, device, preprocess=preprocess)
        H_v = vggt_out["depth"].shape[2]
        W_v = vggt_out["depth"].shape[3]
        init = VGGTInitializer(vggt_out, (H_v, W_v))
        init.to(device)
        return init._state

    windows = _generate_windows(S, window_size, stride)
    print(f"[windowed-vggt] {S} frames → {len(windows)} windows "
          f"(W={window_size}, stride={stride})")

    # ── 1. VGGT forward on each window ───────────────────────────────────────
    window_outputs = []
    for i, (start, end) in enumerate(windows):
        win_paths = image_paths[start:end]
        print(f"[windowed-vggt]   window {i+1:2d}/{len(windows)}: "
              f"frames {start:3d}–{end-1:3d}  ({end-start} frames)")

        with torch.no_grad():
            vggt_out = run_vggt(model, win_paths, device, preprocess=preprocess)

        # Parse raw VGGT output
        pose_enc   = vggt_out["pose_enc"].squeeze(0)           # [Wk, 9]
        depth      = vggt_out["depth"].squeeze(0).squeeze(-1)  # [Wk, H, Wi]
        depth_conf = vggt_out["depth_conf"].squeeze(0)         # [Wk, H, Wi]
        pts3d      = vggt_out["world_points"].squeeze(0)       # [Wk, H, Wi, 3]
        conf       = vggt_out["world_points_conf"].squeeze(0)  # [Wk, H, Wi]

        H_w, W_w = depth.shape[1], depth.shape[2]
        R, T, K  = decode_pose_enc(pose_enc, (H_w, W_w))      # [Wk,3,3], [Wk,3], [Wk,3,3]

        window_outputs.append({
            "start": start, "end": end,
            "R": R, "T": T, "K": K,
            "depth": depth, "depth_conf": depth_conf,
            "pts3d": pts3d, "conf": conf,
        })
        torch.cuda.empty_cache()

    # ── 2. Pose graph optimization ────────────────────────────────────────────
    print("[windowed-vggt] solving pose graph ...")
    alignments = _solve_pose_graph(window_outputs)

    # ── 3. Merge ─────────────────────────────────────────────────────────────
    print("[windowed-vggt] merging window outputs ...")
    merged = _merge_outputs(window_outputs, alignments, S, device)
    print("[windowed-vggt] done.")
    return merged
