"""Low-precision LayerNorm from MosaicML LLM Foundry.

Forces LayerNorm to run in reduced precision (bf16/fp16) instead of
allowing PyTorch autocast to upcast to fp32.

Reference:
https://github.com/mosaicml/llm-foundry/blob/main/llmfoundry/models/layers/norm.py
"""

import torch
import torch.nn as nn


def _cast_if_autocast_enabled(tensor: torch.Tensor) -> torch.Tensor:
    """Cast tensor to autocast dtype if autocast is enabled.

    Args:
        tensor: Input tensor

    Returns:
        Tensor cast to autocast dtype (bf16/fp16) if enabled, otherwise unchanged
    """
    if torch.is_autocast_enabled():
        if tensor.device.type == "cuda":
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            dtype = tensor.dtype
        return tensor.to(dtype=dtype)
    return tensor


class LPLayerNorm(nn.LayerNorm):
    """Low-precision LayerNorm that stays in bf16/fp16.

    Standard LayerNorm gets upcast to fp32 inside autocast for numerical
    stability. This version forces it to stay in reduced precision for speed.

    This is safe when:
    - Using bf16 (better dynamic range than fp16)
    - Using appropriate eps value (1e-5 is usually fine)
    - Gradients are stable
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LayerNorm in reduced precision.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        # Downcast to bf16/fp16 if autocast is enabled
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = (
            _cast_if_autocast_enabled(self.weight)
            if self.weight is not None
            else self.weight
        )
        downcast_bias = (
            _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        )

        # Disable autocast and compute in reduced precision
        with torch.autocast(enabled=False, device_type=x.device.type):
            return torch.nn.functional.layer_norm(
                downcast_x,
                self.normalized_shape,
                downcast_weight,
                downcast_bias,
                self.eps,
            )
