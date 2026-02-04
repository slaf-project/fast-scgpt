"""Device detection and management for Fast-scGPT.

Supports MPS (Apple Silicon), CUDA, and CPU backends.
MPS is prioritized for local development on Mac.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


def get_device() -> torch.device:
    """Get the optimal device for training.

    Priority: CUDA > MPS > CPU

    For production training, CUDA is preferred.
    For local development on Mac, MPS provides GPU acceleration.

    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict[str, object]:
    """Get comprehensive device information for debugging.

    Returns:
        dict: Device information including:
            - device: The selected device
            - device_type: String name of device type
            - cuda_available: Whether CUDA is available
            - mps_available: Whether MPS is available
            - cuda_device_count: Number of CUDA devices (if available)
            - cuda_device_name: Name of first CUDA device (if available)
    """
    device = get_device()

    info: dict[str, object] = {
        "device": device,
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)

    return info


def supports_flash_attention() -> bool:
    """Check if Flash Attention is supported on the current device.

    Flash Attention requires:
    - CUDA with compute capability >= 8.0 (Ampere+)
    - flash-attn package installed

    Returns:
        bool: True if Flash Attention is available.
    """
    if not torch.cuda.is_available():
        return False

    # Check compute capability (need Ampere or newer)
    major, minor = torch.cuda.get_device_capability(0)
    if major < 8:
        return False

    # Check if flash-attn is installed
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def supports_compile() -> bool:
    """Check if torch.compile is supported.

    torch.compile works best on CUDA with recent PyTorch.
    MPS has limited compile support.

    Returns:
        bool: True if torch.compile is recommended.
    """
    # torch.compile works on CUDA and CPU, limited on MPS
    device = get_device()
    if device.type == "mps":
        # MPS compile support is experimental
        return False
    return True


def get_dtype(device: torch.device | None = None) -> torch.dtype:
    """Get the recommended dtype for the given device.

    - CUDA: bfloat16 (if supported) or float16
    - MPS: float32 (bfloat16 not fully supported)
    - CPU: float32

    Args:
        device: Device to check. Uses default device if None.

    Returns:
        torch.dtype: Recommended dtype for training.
    """
    if device is None:
        device = get_device()

    if device.type == "cuda":
        # Prefer bfloat16 on Ampere+ for better numerical stability
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif device.type == "mps":
        # MPS has limited bfloat16 support, use float32 for stability
        return torch.float32
    else:
        return torch.float32
