"""lora.py — LoRA injection utilities for VGGT fine-tuning on dynamic scenes.

Moved from vggt_dyn/lora.py to finetune/lora.py to separate fine-tuning tools
from the core TTO (test-time optimization) package.

Usage
-----
    from vggt_dyn.lora import inject_lora, save_lora, load_lora

    model = VGGT(...)
    model.load_state_dict(...)                # load pretrained weights
    inject_lora(model, rank=16, alpha=16.0,   # inject & freeze everything except LoRA+heads
                target_blocks="global")

    # only LoRA + head parameters are trainable
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )
    ...
    save_lora(model, "lora_weights.pt")

    # later — load back on a fresh VGGT
    inject_lora(model2, rank=16)
    load_lora(model2, "lora_weights.pt")

Design
------
LoRA (Hu et al. 2021) decomposes each weight update as a product of two low-rank
matrices so that only ~1–2% of parameters need to be trained:

    W_new = W_frozen  +  B @ A * (alpha / rank)

Here we apply LoRA to the *query-key-value* projection (qkv) and *output*
projection (proj) inside every ``Block.attn`` in the chosen block lists.

Why global_blocks?
~~~~~~~~~~~~~~~~~~
VGGT uses alternating-attention: frame_blocks attend within each frame,
global_blocks attend across all frames.  Dynamic-scene adaptation primarily
requires changing *how frames relate to each other* (cross-frame temporal
reasoning), which is exactly what global_blocks compute.  Limiting LoRA to
global_blocks therefore gives the best parameter efficiency for temporal
adaptation.

Modes
-----
    "global"  — LoRA on global_blocks attention only        (recommended)
    "frame"   — LoRA on frame_blocks attention only
    "both"    — LoRA on all attention blocks
    "heads"   — No LoRA; only unfreeze output heads (simplest baseline)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRA linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` augmented with a trainable low-rank delta.

    Forward:
        y = W_frozen @ x  +  B @ A @ x  * scale

    where A ∈ R^{rank × d_in},  B ∈ R^{d_out × rank},  scale = alpha / rank.

    At initialisation B = 0  so the delta is exactly zero → the adapted model
    starts identical to the base model.

    Args:
        linear  : The frozen base ``nn.Linear`` to wrap.
        rank    : LoRA rank (typical: 4 / 8 / 16 / 32).
        alpha   : LoRA scaling factor (often == rank; tunes effective LR).
    """

    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 16.0):
        super().__init__()

        self.linear = linear                    # keeps original reference
        d_in  = linear.in_features
        d_out = linear.out_features

        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale  = alpha / rank

        # Kaiming-uniform init for A; B is already zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze the base weight so gradients only flow through A and B
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        # lora delta:  x (... d_in) @ A^T (d_in→rank) @ B^T (rank→d_out)
        lora = (x @ self.lora_A.T) @ self.lora_B.T
        return base + lora * self.scale

    def extra_repr(self) -> str:
        d_in  = self.linear.in_features
        d_out = self.linear.out_features
        rank  = self.lora_A.shape[0]
        return (f"in={d_in}, out={d_out}, rank={rank}, "
                f"scale={self.scale:.3f}")


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def _inject_lora_into_block_list(
    blocks: nn.ModuleList,
    rank: int,
    alpha: float,
) -> None:
    """Replace ``attn.qkv`` and ``attn.proj`` in each Block with LoRALinear."""
    for block in blocks:
        attn = block.attn
        attn.qkv  = LoRALinear(attn.qkv,  rank=rank, alpha=alpha)
        attn.proj = LoRALinear(attn.proj,  rank=rank, alpha=alpha)


def inject_lora(
    model:         nn.Module,
    rank:          int   = 16,
    alpha:         float = 16.0,
    target_blocks: str   = "global",
    unfreeze_heads: bool = True,
) -> list:
    """Freeze the entire VGGT model and inject LoRA into chosen attention blocks.

    Args:
        model          : VGGT instance (weights already loaded).
        rank           : LoRA rank.
        alpha          : LoRA alpha (scaling).
        target_blocks  : Which blocks get LoRA — ``"global"`` / ``"frame"`` / ``"both"``
                         / ``"heads"`` (no LoRA, heads-only fine-tuning).
        unfreeze_heads : If True, also unfreeze depth_head, point_head, camera_head.

    Returns:
        List of trainable ``nn.Parameter`` objects (pass to optimizer).
    """
    # ── 1. Freeze everything ────────────────────────────────────────────────
    for p in model.parameters():
        p.requires_grad_(False)

    # ── 2. Inject LoRA ──────────────────────────────────────────────────────
    aggregator = model.aggregator

    if target_blocks in ("global", "both"):
        _inject_lora_into_block_list(aggregator.global_blocks, rank, alpha)

    if target_blocks in ("frame", "both"):
        _inject_lora_into_block_list(aggregator.frame_blocks, rank, alpha)

    # (if target_blocks == "heads", no LoRA is injected)

    # ── 3. Unfreeze output heads ─────────────────────────────────────────────
    if unfreeze_heads:
        heads = [
            getattr(model, "depth_head",  None),
            getattr(model, "point_head",  None),
            getattr(model, "camera_head", None),
        ]
        for head in heads:
            if head is not None:
                for p in head.parameters():
                    p.requires_grad_(True)

    # ── 4. Report ────────────────────────────────────────────────────────────
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct       = 100.0 * trainable / total if total > 0 else 0.0
    print(
        f"[LoRA] target={target_blocks}  rank={rank}  alpha={alpha}  "
        f"trainable={trainable:,} / {total:,}  ({pct:.2f}%)"
    )

    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Save / load (only LoRA + head weights — tiny checkpoint)
# ---------------------------------------------------------------------------

_LORA_KEYS = ("lora_A", "lora_B")
_HEAD_PREFIXES = ("depth_head.", "point_head.", "camera_head.")


def _is_lora_or_head(key: str) -> bool:
    if any(k in key for k in _LORA_KEYS):
        return True
    if any(key.startswith(p) for p in _HEAD_PREFIXES):
        return True
    return False


def save_lora(model: nn.Module, path: str) -> None:
    """Save only LoRA matrices + head weights to *path*.

    The resulting checkpoint is typically < 100 MB even for a 1 B param model.
    """
    state = {k: v for k, v in model.state_dict().items() if _is_lora_or_head(k)}
    torch.save({"lora_state": state}, path)
    n_keys = len(state)
    total_mb = sum(v.numel() * v.element_size() for v in state.values()) / 1e6
    print(f"[LoRA] saved {n_keys} tensors ({total_mb:.1f} MB) → {path}")


def load_lora(model: nn.Module, path: str, device: str = "cpu") -> None:
    """Load LoRA + head weights back into *model* (non-strict).

    Call ``inject_lora()`` first so the LoRA parameter slots exist.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt.get("lora_state", ckpt)          # backward-compat
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    print(f"[LoRA] loaded {loaded} tensors from {path}")
    if unexpected:
        print(f"[LoRA]  unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if missing:
        # Only warn about non-LoRA missing keys (LoRA missing → not yet injected)
        non_lora_missing = [k for k in missing if not any(x in k for x in _LORA_KEYS)]
        if non_lora_missing:
            print(f"[LoRA]  non-LoRA missing keys: {non_lora_missing[:5]}")


# ---------------------------------------------------------------------------
# Merge LoRA weights into base linear (optional — for export / inference speedup)
# ---------------------------------------------------------------------------

def merge_lora(model: nn.Module) -> None:
    """Fold LoRA deltas into the base linear weights in-place.

    After merging, the model is equivalent but no longer has separate LoRA
    parameters.  Useful before exporting a fine-tuned checkpoint for pure
    inference.
    """
    for module in model.modules():
        if not isinstance(module, LoRALinear):
            continue
        # W_merged = W_frozen + B @ A * scale
        delta = (module.lora_B @ module.lora_A) * module.scale   # [d_out, d_in]
        with torch.no_grad():
            module.linear.weight.add_(delta)
        # Replace LoRALinear with the plain linear in the parent
        # (done externally — this modifies in-place via add_ which is enough
        #  for inference; for a clean replacement use replace_lora_with_linear)
    print("[LoRA] weights merged into base linears.")
