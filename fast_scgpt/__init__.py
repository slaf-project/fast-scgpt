"""Fast-scGPT: Fast reimplementation of scGPT with modded-nanogpt innovations."""

__version__ = "0.1.0"

from fast_scgpt.config import ModelConfig
from fast_scgpt.device import get_device, get_device_info
from fast_scgpt.model import ScGPT

__all__ = [
    "ModelConfig",
    "ScGPT",
    "get_device",
    "get_device_info",
]
