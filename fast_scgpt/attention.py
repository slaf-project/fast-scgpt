"""Flash Attention integration with SDPA fallback.

Flash Attention 2 provides memory-efficient attention that scales O(n) instead of O(n²),
enabling much longer sequences and larger batch sizes.

Requires: pip install flash-attn (CUDA only, sm80+)
"""

import torch
import torch.nn.functional as F
from loguru import logger

# Try to import Flash Attention 2
FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_ERROR: str | None = None
try:
    from flash_attn import flash_attn_func

    FLASH_ATTN_AVAILABLE = True
except ImportError as e:
    flash_attn_func = None
    FLASH_ATTN_ERROR = str(e)
except Exception as e:
    flash_attn_func = None
    FLASH_ATTN_ERROR = f"{type(e).__name__}: {e}"


def check_flash_attn() -> bool:
    """Check if Flash Attention is available and log status."""
    if FLASH_ATTN_AVAILABLE:
        logger.info("Flash Attention 2 is available")
        return True
    else:
        logger.warning("Flash Attention not available, using PyTorch SDPA fallback")
        if FLASH_ATTN_ERROR:
            logger.warning(f"Flash Attention import error: {FLASH_ATTN_ERROR}")
        return False


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Memory-efficient attention with Flash Attention 2 and SDPA fallback.

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
    if FLASH_ATTN_AVAILABLE and q.is_cuda and attn_mask is None:
        # Flash Attention 2 expects (batch, seqlen, nheads, headdim)
        # and returns same shape
        result: torch.Tensor = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p if q.requires_grad else 0.0,
            causal=causal,
            softmax_scale=scale,
        )
        return result
    else:
        # PyTorch SDPA fallback
        # SDPA expects (batch, nheads, seqlen, headdim)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if q.requires_grad else 0.0,
            is_causal=causal,
            scale=scale,
        )


def attention_with_reshape(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention that handles tensor format differences between FA2 and SDPA.

    Input tensors should be in SDPA format: (batch, nheads, seqlen, headdim)
    This function will transpose for FA2 if needed.

    Args:
        q, k, v: Tensors of shape (batch, nheads, seqlen, headdim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        causal: If True, apply causal masking
        scale: Optional scale factor

    Returns:
        Output tensor of shape (batch, nheads, seqlen, headdim)
    """
    if FLASH_ATTN_AVAILABLE and q.is_cuda and attn_mask is None:
        # Transpose from (batch, nheads, seqlen, headdim) to (batch, seqlen, nheads, headdim)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        out: torch.Tensor = flash_attn_func(
            q,
            k,
            v,
            dropout_p=dropout_p if q.requires_grad else 0.0,
            causal=causal,
            softmax_scale=scale,
        )

        # Transpose back to (batch, nheads, seqlen, headdim)
        return out.transpose(1, 2)
    else:
        # SDPA already expects (batch, nheads, seqlen, headdim)
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p if q.requires_grad else 0.0,
            is_causal=causal,
            scale=scale,
        )
