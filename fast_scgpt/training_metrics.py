"""Compute MFU, achieved TFLOPS, throughput and other GPU utilization metrics for training runs."""

from __future__ import annotations


def get_param_count(model_size: str) -> int:
    """Return parameter count for the given model size (small/scgpt/base/large)."""
    from fast_scgpt.config import ModelConfig
    from fast_scgpt.model import ScGPT

    if model_size == "small":
        config = ModelConfig.small()
    elif model_size == "scgpt":
        config = ModelConfig.scgpt_matched()
    elif model_size == "base":
        config = ModelConfig.base()
    elif model_size == "large":
        config = ModelConfig.large()
    else:
        raise ValueError(f"Unknown model_size: {model_size}")
    model = ScGPT(config)
    return sum(p.numel() for p in model.parameters())


# Peak TFLOPS (FP16/BF16 dense) for common GPUs. Key by substring match on gpu_name.
GPU_PEAK_TFLOPS: dict[str, float] = {
    "H100": 989.0,  # H100 SXM
    "A100": 312.0,  # A100 80GB
    "A10": 125.0,  # A10G
    "L4": 121.0,  # L4
}


def _peak_tflops_for_gpu(gpu_name: str) -> float:
    for key, tflops in GPU_PEAK_TFLOPS.items():
        if key in gpu_name:
            return tflops
    return 312.0  # fallback to A100-like


def flops_per_step_theoretical(
    param_count: int,
    seq_len: int,
    batch_size: int,
) -> float:
    """Approximate FLOPs per training step (forward + backward) for a transformer.

    Uses the common 6 * N * seq * batch for forward and ~2x for backward (12 total).
    """
    return 12.0 * float(param_count) * seq_len * batch_size


def compute_training_metrics(summary: dict) -> dict:
    """Compute MFU, achieved TFLOPS, throughput and steps/sec from a training summary.

    Expects summary to contain at least:
      - status, elapsed_sec, n_steps, effective_batch_size, num_gpus
      - max_genes, model_size, gpu_name

    Optionally from train_ddp metrics file: peak_memory_gb, memory_utilization_pct,
    gpu_utilization_pct, sm_efficiency_pct (nvidia-smi / dmon), training_elapsed_sec.
    When ``training_elapsed_sec`` is present we use it for MFU and achieved TFLOPS
    (wall time for all optimizer steps, including a slow first step). Throughput and
    ``steps_per_sec`` use ``median_cells_per_sec`` / ``median_step_time_ms`` when
    provided so benchmark printouts match the training summary (median step, first
    step omitted from median).
    """
    if summary.get("status") != "success":
        return {}

    # Use training time (actual compute) when available so throughput/MFU match train_ddp
    elapsed = summary.get("training_elapsed_sec") or summary["elapsed_sec"]
    n_steps = summary["n_steps"]
    effective_batch = summary["effective_batch_size"]
    num_gpus = summary["num_gpus"]
    max_genes = summary["max_genes"]
    model_size = summary["model_size"]
    gpu_name = summary["gpu_name"]

    if elapsed <= 0 or n_steps <= 0:
        return {}

    # Sequence length: dual-stream scGPT uses max_genes + 2 (CLS + SEP).
    seq_len = max_genes + 2
    param_count = get_param_count(model_size)
    flops_per_step = flops_per_step_theoretical(param_count, seq_len, effective_batch)
    total_flops = flops_per_step * n_steps
    peak_tflops_per_gpu = _peak_tflops_for_gpu(gpu_name)
    # Peak FLOPs/s for the whole run = num_gpus * peak_tflops_per_gpu (in TFLOPS) * 1e12
    peak_flops_per_sec = num_gpus * peak_tflops_per_gpu * 1e12

    achieved_tflops_total = total_flops / (elapsed * 1e12)
    achieved_tflops_per_gpu = achieved_tflops_total / num_gpus
    mfu_pct = (
        100.0 * (total_flops / (elapsed * peak_flops_per_sec))
        if peak_flops_per_sec > 0
        else 0.0
    )
    throughput_cells_per_sec = (effective_batch * n_steps) / elapsed
    steps_per_sec = n_steps / elapsed

    median_cells = summary.get("median_cells_per_sec")
    median_step_ms = summary.get("median_step_time_ms")
    if median_cells is not None and float(median_cells) > 0:
        throughput_cells_per_sec = float(median_cells)
    if median_step_ms is not None and float(median_step_ms) > 0:
        steps_per_sec = 1000.0 / float(median_step_ms)

    out = {
        "mfu_pct": round(mfu_pct, 2),
        "achieved_tflops_total": round(achieved_tflops_total, 1),
        "achieved_tflops_per_gpu": round(achieved_tflops_per_gpu, 1),
        "throughput_cells_per_sec": round(throughput_cells_per_sec, 1),
        "steps_per_sec": round(steps_per_sec, 2),
    }
    if "peak_memory_gb" in summary:
        out["peak_memory_gb"] = summary["peak_memory_gb"]
    if "memory_utilization_pct" in summary:
        out["memory_utilization_pct"] = summary["memory_utilization_pct"]
    if "gpu_utilization_pct" in summary:
        out["gpu_utilization_pct"] = summary["gpu_utilization_pct"]
    if "sm_efficiency_pct" in summary:
        out["sm_efficiency_pct"] = summary["sm_efficiency_pct"]
    return out
