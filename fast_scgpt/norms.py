"""Normalization layers for fast-scGPT.

RMSNorm (Root Mean Square Layer Normalization) is faster than LayerNorm
and doesn't require learnable parameters, following modded-nanogpt design.
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm:
    - Only normalizes by RMS (no mean centering)
    - Has no learnable gamma/beta parameters
    - Is computationally cheaper

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize RMSNorm.

        Args:
            dim: Hidden dimension (unused, kept for API compatibility)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., dim)

        Returns:
            Normalized tensor of same shape
        """
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms
