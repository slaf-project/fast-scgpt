"""Attention module with Flash Attention 2/3, optional FA4 (CuTe), and SDPA fallback.

Backend is selected at import time via env ``FAST_SCGPT_FLASH_ATTN_BACKEND``:

- ``fa3`` (default): ``flash-attn`` wheel — ``flash_attn.flash_attn_func`` (FA2 on Ampere, FA3 on Hopper).
- ``fa4``: ``flash-attn-4`` — ``flash_attn.cute.flash_attn_func`` (Hopper/Blackwell).
- ``sdpa``: skip packaged FlashAttention; use PyTorch SDPA only.

On H100, the ``fa3`` wheel uses native (B, T, H, D) layout without transpose overhead when FA3 is active.

FA4's ``flash_attn_func`` does not implement attention dropout; when ``dropout_p > 0`` we apply
``F.dropout`` on the attention output (approximation vs softmax dropout for fair A/B timing).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from loguru import logger

_FLASH_BACKEND = os.environ.get("FAST_SCGPT_FLASH_ATTN_BACKEND", "fa3").strip().lower()
if _FLASH_BACKEND not in ("fa3", "fa4", "sdpa"):
    _FLASH_BACKEND = "fa3"

FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_ERROR: str | None = None
# "fa3" | "fa4" | None (sdpa-only or unavailable)
FLASH_ATTN_KIND: str | None = None
flash_attn_func: Callable[..., torch.Tensor] | None = None
_flash_attn_func_fa4: Callable[..., object] | None = None

if _FLASH_BACKEND == "sdpa":
    pass  # SDPA only
elif _FLASH_BACKEND == "fa4":
    try:
        from flash_attn.cute import flash_attn_func as _flash_attn_cute_impl

        _flash_attn_func_fa4 = _flash_attn_cute_impl
        FLASH_ATTN_AVAILABLE = True
        FLASH_ATTN_KIND = "fa4"
    except ImportError as e:
        FLASH_ATTN_ERROR = str(e)
    except Exception as e:
        FLASH_ATTN_ERROR = f"{type(e).__name__}: {e}"
else:
    try:
        from flash_attn import flash_attn_func as _flash_fa3_wheel

        flash_attn_func = _flash_fa3_wheel
        FLASH_ATTN_AVAILABLE = True
        FLASH_ATTN_KIND = "fa3"
    except ImportError as e:
        FLASH_ATTN_ERROR = str(e)
    except Exception as e:
        FLASH_ATTN_ERROR = f"{type(e).__name__}: {e}"


def attention_backend_label() -> str:
    """Human-readable backend for logs and benchmark JSON."""
    if _FLASH_BACKEND == "sdpa":
        return "sdpa"
    if FLASH_ATTN_KIND == "fa4":
        return "fa4"
    if FLASH_ATTN_KIND == "fa3":
        return "fa3"
    if _FLASH_BACKEND == "fa4":
        return "fa4_import_failed_sdpa"
    if _FLASH_BACKEND == "fa3":
        return "fa3_import_failed_sdpa"
    return "sdpa_fallback"


def _call_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    causal: bool,
    scale: float | None,
) -> torch.Tensor:
    """Dispatch to flash-attn wheel, FA4, or raise if misconfigured."""
    if FLASH_ATTN_KIND == "fa3" and flash_attn_func is not None:
        return cast(
            torch.Tensor,
            flash_attn_func(
                q,
                k,
                v,
                dropout_p=dropout_p,
                causal=causal,
                softmax_scale=scale,
            ),
        )
    if FLASH_ATTN_KIND == "fa4" and _flash_attn_func_fa4 is not None:
        fa_ret = _flash_attn_func_fa4(
            q,
            k,
            v,
            softmax_scale=scale,
            causal=causal,
            return_lse=False,
        )
        # FA4's autograd.Function forward returns (out, lse); unpack for dropout / callers.
        raw = fa_ret[0] if isinstance(fa_ret, tuple) else fa_ret
        out_t = cast(torch.Tensor, raw)
        if dropout_p > 0.0:
            out_t = F.dropout(out_t, p=dropout_p, training=True)
        return out_t
    raise RuntimeError("Flash attention dispatch called with no active backend")


def is_hopper_gpu() -> bool:
    """Check if running on H100 (Hopper architecture, sm90)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major >= 9


def check_flash_attn() -> bool:
    """Check attention backend status and log."""
    import torch.backends.cuda

    hopper = is_hopper_gpu()
    if hopper:
        logger.info(
            "Detected H100 (Hopper) GPU - native (B,T,H,D) layout when FlashAttention is active"
        )

    if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
        logger.info(
            f"SDPA flash backend available: {torch.backends.cuda.flash_sdp_enabled()}"
        )
        logger.info(
            f"SDPA mem_efficient backend available: {torch.backends.cuda.mem_efficient_sdp_enabled()}"
        )

    logger.info(
        f"FAST_SCGPT_FLASH_ATTN_BACKEND={_FLASH_BACKEND!r} -> {attention_backend_label()}"
    )
    if FLASH_ATTN_AVAILABLE:
        if FLASH_ATTN_KIND == "fa4":
            logger.info("Using FlashAttention-4 (flash_attn.cute)")
        elif FLASH_ATTN_KIND == "fa3":
            if hopper:
                logger.info(
                    "FlashAttention (flash-attn wheel) — FA3 path on Hopper when supported"
                )
            else:
                logger.info("FlashAttention (flash-attn wheel) — FA2 on this GPU")
    else:
        logger.info(
            "Using PyTorch SDPA (no packaged FlashAttention for this backend/image)"
        )
        if FLASH_ATTN_ERROR:
            logger.debug(f"flash_attn import note: {FLASH_ATTN_ERROR}")

    return FLASH_ATTN_AVAILABLE or True  # We always have SDPA as fallback


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Memory-efficient attention with Flash Attention and SDPA fallback.

    Args:
        q: Query tensor of shape (batch, seqlen, nheads, headdim) for FA2
           or (batch, nheads, seqlen, headdim) for SDPA
        k: Key tensor, same shape as q
        v: Value tensor, same shape as q
        attn_mask: Optional attention mask (SDPA only, ignored by FA2)
        dropout_p: Dropout probability
        causal: If True, apply causal masking
        scale: Optional scale factor (default: 1/sqrt(headdim))

    Returns:
        Output tensor of same shape as input
    """
    use_flash = (
        FLASH_ATTN_AVAILABLE
        and FLASH_ATTN_KIND is not None
        and q.is_cuda
        and attn_mask is None
    )
    if use_flash:
        dp = dropout_p if q.requires_grad else 0.0
        return _call_flash_attention(q, k, v, dropout_p=dp, causal=causal, scale=scale)
    # SDPA expects (batch, nheads, seqlen, headdim); see docstring
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if q.requires_grad else 0.0,
        is_causal=causal,
        scale=scale,
    )


def attention_native_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention with (B, seqlen, nheads, headdim) layout.

    Uses packaged FlashAttention when available; otherwise transposes for SDPA.

    Args:
        q, k, v: Tensors of shape (batch, seqlen, nheads, headdim)
        dropout_p: Dropout probability
        causal: If True, apply causal masking
        scale: Optional scale factor

    Returns:
        Output tensor of shape (batch, seqlen, nheads, headdim)
    """
    if FLASH_ATTN_AVAILABLE and FLASH_ATTN_KIND is not None and q.is_cuda:
        dp = dropout_p if q.requires_grad else 0.0
        return _call_flash_attention(q, k, v, dropout_p=dp, causal=causal, scale=scale)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q_sdpa,
        k_sdpa,
        v_sdpa,
        dropout_p=dropout_p if q.requires_grad else 0.0,
        is_causal=causal,
        scale=scale,
    )
    return out.transpose(1, 2)


def attention_with_reshape(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention for SDPA format (B, H, T, D).

    DEPRECATED: Use attention_native_layout with (B, T, H, D) format instead
    for better performance on H100.

    Args:
        q, k, v: Tensors of shape (batch, nheads, seqlen, headdim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        causal: If True, apply causal masking
        scale: Optional scale factor

    Returns:
        Output tensor of shape (batch, nheads, seqlen, headdim)
    """
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if q.requires_grad else 0.0,
        is_causal=causal,
        scale=scale,
    )
