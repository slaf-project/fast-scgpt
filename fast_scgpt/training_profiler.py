"""PyTorch profiler helpers for training (forward / backward CUDA breakdown).

Use ``train(..., torch_profiler_steps=N)`` to capture N optimizer steps after warmup.
Module-level labels (``scgpt.*``) are enabled only while the profiler context is active
via env ``FAST_SCGPT_TORCH_PROFILER=1`` so normal runs and compile traces stay clean.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ENV_TORCH_PROFILER = "FAST_SCGPT_TORCH_PROFILER"


def torch_profiler_active() -> bool:
    return os.environ.get(ENV_TORCH_PROFILER, "").lower() in ("1", "true", "yes")


def set_torch_profiler_active(active: bool) -> None:
    if active:
        os.environ[ENV_TORCH_PROFILER] = "1"
    else:
        os.environ.pop(ENV_TORCH_PROFILER, None)


def build_torch_profiler(
    *,
    record_shapes: bool = False,
    with_stack: bool = False,
    profile_memory: bool = False,
) -> Any:
    """Create a ``torch.profiler.profile`` for CUDA + CPU training."""
    import torch

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
        with_flops=False,
    )


def _self_device_time_us(event: Any) -> int:
    """Self time on device (microseconds); API differs across PyTorch versions."""
    # FunctionEventAvg (2.4+): int fields self_device_time_total / device_time_total
    if hasattr(event, "self_device_time_total"):
        return int(event.self_device_time_total)
    v = getattr(event, "self_cuda_time_total", None)
    return int(v) if v is not None else 0


def _device_time_total_us(event: Any) -> int:
    """Inclusive device time (microseconds)."""
    if hasattr(event, "device_time_total"):
        return int(event.device_time_total)
    v = getattr(event, "cuda_time_total", None)
    return int(v) if v is not None else 0


def _key_averages_table(prof: Any, *, row_limit: int) -> str:
    """``EventList.table`` with a ``sort_by`` key that exists on this PyTorch build."""
    events = prof.key_averages()
    for sort_by in ("self_device_time_total", "self_cuda_time_total"):
        try:
            return str(events.table(sort_by=sort_by, row_limit=row_limit))
        except Exception:
            continue
    return str(events.table(row_limit=row_limit))


def _user_annotation_rows(prof: Any) -> list[tuple[str, float, float]]:
    """Return (name, self_device_ms, device_total_ms) for user annotations only."""
    rows: list[tuple[str, float, float]] = []
    for e in prof.key_averages():
        name = str(e.key)
        if not name.startswith("scgpt."):
            continue
        self_us = _self_device_time_us(e)
        tot_us = _device_time_total_us(e)
        rows.append(
            (
                name,
                self_us / 1000.0 if self_us > 0 else 0.0,
                tot_us / 1000.0 if tot_us > 0 else 0.0,
            )
        )
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def format_profiler_report(
    prof: Any,
    *,
    top_ops: int = 30,
    top_annotations: int = 40,
) -> str:
    """Human-readable summary: user regions (scgpt.*) + top device ops by self time."""
    lines: list[str] = []
    lines.append(
        "Torch profiler: user regions (scgpt.*), self device ms (aggregate over captured steps)"
    )
    ann = _user_annotation_rows(prof)
    if ann:
        lines.append(f"{'name':<40} {'self_dev_ms':>14} {'dev_total_ms':>14}")
        for name, self_ms, tot_ms in ann[:top_annotations]:
            lines.append(f"{name:<40} {self_ms:>14.3f} {tot_ms:>14.3f}")
    else:
        lines.append(
            "(no scgpt.* user regions — enable during capture via train(torch_profiler_steps=...))"
        )

    lines.append("")
    lines.append("Torch profiler: top device ops by self time (kernel / aten)")
    lines.append(_key_averages_table(prof, row_limit=top_ops))
    return "\n".join(lines)


def export_chrome_trace(prof: Any, path: str | Path) -> None:
    """Write Chrome trace JSON for chrome://tracing or Perfetto."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path))
