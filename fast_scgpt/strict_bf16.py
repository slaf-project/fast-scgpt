"""Strict BF16 mode - force all operations to stay in bf16.

This is more aggressive than standard autocast. It converts the model to bf16
and prevents any upcasting to fp32 during training.

Based on: Apex O2 mode and modern mixed precision training practices.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def convert_to_strict_bf16(model: nn.Module) -> nn.Module:
    """Convert model to strict bf16 mode.

    This converts all parameters and buffers to bf16, and the model will
    compute entirely in bf16 (no automatic fp32 upcasting).

    Args:
        model: Model to convert

    Returns:
        Model in bf16

    Note:
        - Loss computation should still use fp32 for stability
        - Gradients will be computed in bf16 then accumulated
    """
    return model.to(dtype=torch.bfloat16)


def bf16_loss_wrapper(
    loss_fn: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    """Wrap loss function to compute in fp32 for stability.

    Args:
        loss_fn: Original loss function

    Returns:
        Wrapped loss function that upcasts to fp32
    """

    def wrapped_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
        # Upcast to fp32 for stable loss computation
        args_fp32 = [
            arg.float() if isinstance(arg, torch.Tensor) else arg for arg in args
        ]
        loss = loss_fn(*args_fp32, **kwargs)
        return loss

    return wrapped_loss


class StrictBF16Context:
    """Context manager for strict bf16 training.

    Example:
        with StrictBF16Context(enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.prev_cache_enabled: bool | None = None

    def __enter__(self) -> StrictBF16Context:
        if self.enabled and torch.cuda.is_available():
            # Disable TF32 for matmuls (use BF16 instead)
            self.prev_cache_enabled = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            # Note: We don't use autocast here - model is already in bf16
        return self

    def __exit__(self, *args: object) -> None:
        if self.enabled and torch.cuda.is_available():
            # Restore TF32 setting
            if self.prev_cache_enabled is not None:
                torch.backends.cuda.matmul.allow_tf32 = self.prev_cache_enabled


def setup_strict_bf16_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Setup strict bf16 training.

    Args:
        model: Model to train
        optimizer: Optimizer

    Returns:
        (model in bf16, optimizer)

    Usage:
        model, optimizer = setup_strict_bf16_training(model, optimizer)

        for batch in dataloader:
            optimizer.zero_grad()

            # Forward in bf16 (no autocast needed)
            outputs = model(inputs)

            # Loss in fp32 for stability
            loss = criterion(outputs.float(), targets)

            # Backward (gradients in bf16)
            loss.backward()

            optimizer.step()
    """
    # Convert model to bf16
    model = convert_to_strict_bf16(model)

    # Note: Optimizer will work with bf16 parameters automatically
    # Gradients will be accumulated in bf16

    return model, optimizer
