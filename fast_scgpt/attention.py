"""Attention module with Flash Attention 3 and SDPA support.

On H100 (Hopper), uses Flash Attention 3 with native (B, T, H, D) layout.
On other GPUs, uses PyTorch SDPA which auto-dispatches to best backend.

Flash Attention 3 benefits:
- Native (B, T, H, D) layout - no transpose overhead
- Optimized for Hopper architecture
- ~9% speedup over FA2
"""

import torch
import torch.nn.functional as F
from loguru import logger

# Try to import Flash Attention (FA3 on H100, FA2 on older GPUs)
FLASH_ATTN_AVAILABLE = False
FLASH_ATTN_ERROR: str | None = None
flash_attn_func = None

try:
    from flash_attn import flash_attn_func as _flash_attn_func

    flash_attn_func = _flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError as e:
    FLASH_ATTN_ERROR = str(e)
except Exception as e:
    FLASH_ATTN_ERROR = f"{type(e).__name__}: {e}"


def is_hopper_gpu() -> bool:
    """Check if running on H100 (Hopper architecture, sm90)."""
    if not torch.cuda.is_available():
        return False
    props = torch.cuda.get_device_properties(0)
    # H100 is compute capability 9.0
    return props.major >= 9


def check_flash_attn() -> bool:
    """Check attention backend status and log."""
    import torch.backends.cuda

    hopper = is_hopper_gpu()
    if hopper:
        logger.info(
            "Detected H100 (Hopper) GPU - will use FA3 native layout if available"
        )

    # Log SDPA backend availability
    if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
        logger.info(
            f"SDPA flash backend available: {torch.backends.cuda.flash_sdp_enabled()}"
        )
        logger.info(
            f"SDPA mem_efficient backend available: {torch.backends.cuda.mem_efficient_sdp_enabled()}"
        )

    if FLASH_ATTN_AVAILABLE:
        if hopper:
            logger.info("Flash Attention 3 available - using native (B,T,H,D) layout")
        else:
            logger.info("Flash Attention 2 available")
    else:
        logger.info("Using PyTorch SDPA (auto-selects best backend)")
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
    if (
        FLASH_ATTN_AVAILABLE
        and flash_attn_func is not None
        and q.is_cuda
        and attn_mask is None
    ):
        # Flash Attention 2/3 expects (batch, seqlen, nheads, headdim)
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
    """Attention using PyTorch SDPA.

    SDPA automatically dispatches to the best backend (flash, memory-efficient, or math)
    based on inputs and hardware. On A100, it uses flash attention internally.

    The explicit flash_attn package requires format conversion that adds overhead,
    so we use SDPA which handles this optimally.

    Args:
        q, k, v: Tensors of shape (batch, nheads, seqlen, headdim)
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        causal: If True, apply causal masking
        scale: Optional scale factor

    Returns:
        Output tensor of shape (batch, nheads, seqlen, headdim)
    """
    # SDPA auto-selects best backend (flash/mem-efficient/math)
    # On A100 with no mask, it uses flash attention internally
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p if q.requires_grad else 0.0,
        is_causal=causal,
        scale=scale,
    )
