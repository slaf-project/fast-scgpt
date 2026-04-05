"""Modal training script for fast-scGPT GPU benchmarking.

This script deploys training to Modal's GPU infrastructure for CUDA benchmarking.

Usage:
    # Run with defaults (500 steps)
    modal run modal_train.py

    # Custom configuration
    modal run modal_train.py --batch-size 64 --n-steps 1000

    # Test with minimal config
    modal run modal_train.py --batch-size 8 --max-genes 128 --n-steps 50

    # Run in detached mode (continues after terminal closes)
    modal run --detach modal_train.py --n-steps 1000

    # Check logs for detached runs
    modal app logs fast-scgpt-benchmark

Results are saved to /data/benchmark_results/ on the volume.
"""

import modal

# Create Modal app
app = modal.App("fast-scgpt-benchmark")

# Mount existing Modal volume with SLAF datasets
slaf_volume = modal.Volume.from_name("slaf-datasets")

# Build image with CUDA + dependencies
# Flash Attention 3 on H100 uses native (B,T,H,D) layout - no transpose overhead
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.4.0,<2.5.0",  # Pin for flash-attn compatibility
        "einops>=0.7",
        "numpy>=1.24",
        "loguru>=0.7",
        "polars>=0.20",
        "pyarrow>=14.0",
        "psutil>=5.9",
        "s3fs>=2024.2",
        "packaging",
        "ninja",
        "slafdb",
    )
    # Flash Attention - FA3 on H100, FA2 on A100
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    # Copy local fast_scgpt package
    .add_local_dir("fast_scgpt", "/root/fast_scgpt")
)


@app.function(
    image=image,
    gpu="H100",
    timeout=7200,  # 2 hours max
    volumes={"/data": slaf_volume},
    secrets=[modal.Secret.from_name("s3-credentials")],
)
def train_on_modal(
    batch_size: int = 32,
    max_genes: int = 64,
    n_steps: int = 500,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    model_size: str = "small",
    gradient_accumulation_steps: int = 1,
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    use_swiglu: bool = False,
    use_lp_layernorm: bool = False,
    use_softcap: bool = False,
    use_strict_bf16: bool = False,
    profile: bool = False,
    data_source: str = "s3",
) -> dict:
    """Run training with GPU metrics on Modal.

    Args:
        batch_size: Batch size for training (micro-batch if using accumulation)
        max_genes: Maximum genes per cell (affects seq_len)
        n_steps: Number of training steps
        learning_rate: Learning rate
        log_every: Log interval
        model_size: Model size preset (small/scgpt/base/large)
            - small: 4L, 4H, 256D (dev/testing)
            - scgpt: 12L, 4H, 512D, 1×FF (~51M params, matches original scGPT)
            - base: 12L, 8H, 512D, 4×FF (~102M params, standard transformer)
            - large: 24L, 16H, 1024D, 4×FF (scaling experiments)
        gradient_accumulation_steps: Accumulate gradients over N micro-batches
            Effective batch = batch_size * gradient_accumulation_steps
        use_gradient_checkpointing: Trade compute for ~50% activation memory savings
        use_compile: Use torch.compile for fused kernels (may speed up training)
        compile_mode: "reduce-overhead" | "max-autotune" | "default" (for MFU try max-autotune)
        use_swiglu: Use SwiGLU activation (Llama-style) instead of GELU
        use_lp_layernorm: Force LayerNorm to stay in bf16 (Tahoe-X1 optimization)
        use_softcap: Apply logit softcapping to prevent extreme logits (nanochat)
        use_strict_bf16: Convert entire model to bf16 (more aggressive than autocast)
        profile: Log timing breakdown (data/mask/forward/backward/optim)
        data_source: Data source - "s3", "volume", or "hf" (HuggingFace)

    Returns:
        dict with training summary metrics
    """
    import sys

    sys.path.insert(0, "/root")

    import torch
    from loguru import logger

    from fast_scgpt.config import ModelConfig
    from fast_scgpt.train import train

    # Log GPU info
    logger.info("=" * 60)
    logger.info("Modal GPU Benchmark - fast-scGPT")
    logger.info("=" * 60)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
        logger.info(f"Compute Capability: {props.major}.{props.minor}")
        logger.info(f"SM Count: {props.multi_processor_count}")
    else:
        logger.error("CUDA not available! Check Modal GPU allocation.")
        return {"error": "CUDA not available"}

    # Select data source
    if data_source == "volume":
        slaf_path = "/data/tigris/Tahoe100M_train_SLAF"
    elif data_source == "hf":
        slaf_path = "hf://datasets/slaf-project/Tahoe-100M/data/train"
    elif data_source == "s3":
        slaf_path = "s3://slaf-datasets/Tahoe100M_train_SLAF"
    else:
        logger.error(f"Unknown data_source: {data_source}. Use 's3', 'volume', or 'hf'")
        return {"error": f"Unknown data_source: {data_source}"}

    logger.info(f"Data source: {data_source}")
    logger.info(f"SLAF path: {slaf_path}")

    # Verify S3 connectivity before creating dataloader
    import os

    if slaf_path.startswith("s3://"):
        logger.info("Verifying S3 connectivity...")
        try:
            import s3fs

            # Check if AWS credentials are available
            aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
            aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            logger.info(f"AWS_ACCESS_KEY_ID present: {bool(aws_key)}")
            logger.info(f"AWS_DEFAULT_REGION: {aws_region}")

            fs = s3fs.S3FileSystem()
            # List files in the SLAF directory
            bucket_path = slaf_path.replace("s3://", "")
            files = fs.ls(bucket_path)
            logger.info(f"S3 path accessible, found {len(files)} items:")
            for f in files[:5]:  # Show first 5
                logger.info(f"  {f}")
            if len(files) > 5:
                logger.info(f"  ... and {len(files) - 5} more")
        except Exception as e:
            logger.error(f"S3 connectivity check failed: {e}")
            return {"error": f"S3 access failed: {e}"}

    # Get model config
    if model_size == "small":
        config = ModelConfig.small()
    elif model_size == "scgpt":
        config = ModelConfig.scgpt_matched()
    elif model_size == "base":
        config = ModelConfig.base()
    elif model_size == "large":
        config = ModelConfig.large()
    else:
        logger.error(f"Unknown model size: {model_size}")
        return {"error": f"Unknown model size: {model_size}"}

    # Apply optimizations if requested
    if use_swiglu:
        config.use_swiglu = True
        logger.info("Using SwiGLU activation (Llama-style)")

    if use_lp_layernorm:
        config.use_lp_layernorm = True
        logger.info("Using low-precision LayerNorm (Tahoe-X1 style)")

    if use_softcap:
        config.use_softcap = True
        logger.info("Using logit softcapping (nanochat optimization)")

    logger.info(f"Model size: {model_size}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max genes: {max_genes}")
    logger.info(f"N steps: {n_steps}")
    logger.info(f"Config vocab_size: {config.vocab_size}")
    logger.info(f"Config n_expression_bins: {config.n_expression_bins}")
    logger.info(f"Config total_vocab_size: {config.total_vocab_size}")
    logger.info("=" * 60)

    # Enable sync CUDA errors for better debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Memory optimization: expandable segments reduces fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Run training
    import time

    start_time = time.time()
    metrics = train(
        slaf_path=slaf_path,
        config=config,
        n_steps=n_steps,
        batch_size=batch_size,
        max_genes=max_genes,
        learning_rate=learning_rate,
        log_every=log_every,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_compile=use_compile,
        compile_mode=compile_mode,
        profile=profile,
        use_strict_bf16=use_strict_bf16,
    )
    elapsed_sec = time.time() - start_time

    # Build results dict (median excludes first batch warmup)
    summary = metrics.summary()
    median_ms = summary["median_step_time_ms"]
    effective_batch = batch_size * gradient_accumulation_steps
    training_elapsed_sec = sum(metrics._step_times) / 1000.0

    result = {
        "status": "success",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "effective_batch_size": effective_batch,
        "max_genes": max_genes,
        "model_size": model_size,
        "elapsed_sec": round(elapsed_sec, 1),
        "training_elapsed_sec": round(training_elapsed_sec, 2),
        "num_gpus": 1,
        "avg_step_time_ms": summary["avg_step_time_ms"],
        "median_step_time_ms": median_ms,
        "median_cells_per_sec": (
            batch_size / (median_ms / 1000) if median_ms > 0 else 0
        ),
        "total_cells": summary["total_cells"],
        "total_tokens": summary["total_tokens"],
        "peak_memory_gb": summary["peak_memory_gb"],
        "memory_utilization_pct": summary["memory_utilization_pct"],
        "gpu_name": torch.cuda.get_device_name(0),
    }
    if "gpu_utilization_pct" in summary:
        result["gpu_utilization_pct"] = summary["gpu_utilization_pct"]
    if "sm_efficiency_pct" in summary:
        result["sm_efficiency_pct"] = summary["sm_efficiency_pct"]

    # MFU, achieved TFLOPS, throughput, steps/sec (comparable to distributed benchmarks)
    from fast_scgpt.training_metrics import compute_training_metrics

    result.update(compute_training_metrics(result))

    # Save results to volume for detached runs
    import json
    from datetime import datetime

    results_dir = "/data/benchmark_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{results_dir}/benchmark_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    return result


@app.local_entrypoint()
def main(
    batch_size: int = 32,
    max_genes: int = 64,
    n_steps: int = 500,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    model_size: str = "small",
    gradient_accumulation_steps: int = 1,
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    use_swiglu: bool = False,
    use_lp_layernorm: bool = False,
    use_softcap: bool = False,
    use_strict_bf16: bool = False,
    profile: bool = False,
    data_source: str = "s3",
) -> None:
    """Run training benchmark from local machine.

    Args:
        batch_size: Batch size (default: 32)
        max_genes: Max genes per cell (default: 64)
        n_steps: Training steps (default: 500)
        learning_rate: LR (default: 1e-4)
        log_every: Log interval (default: 1)
        model_size: small/scgpt/base/large (default: small)
        gradient_accumulation_steps: Effective batch = batch_size * this
        use_gradient_checkpointing: Trade compute for memory
        use_compile: Use torch.compile for fused kernels
        compile_mode: reduce-overhead (default) or max-autotune for better MFU
        use_swiglu: Use SwiGLU activation instead of GELU
        use_lp_layernorm: Force LayerNorm to stay in bf16
        use_softcap: Apply logit softcapping (nanochat)
        profile: Log timing breakdown per step
        data_source: Data source - "s3", "volume", or "hf"
    """
    effective_batch = batch_size * gradient_accumulation_steps
    print("Launching fast-scGPT training on Modal GPU...")
    print(f"  batch_size={batch_size}, max_genes={max_genes}, n_steps={n_steps}")
    print(f"  model_size={model_size}, lr={learning_rate}")
    print(f"  gradient_accumulation_steps={gradient_accumulation_steps}")
    print(f"  effective_batch_size={effective_batch}")
    print(f"  use_gradient_checkpointing={use_gradient_checkpointing}")
    print(f"  use_compile={use_compile}")
    print(f"  compile_mode={compile_mode}")
    print(f"  use_swiglu={use_swiglu}")
    print(f"  use_lp_layernorm={use_lp_layernorm}")
    print(f"  use_softcap={use_softcap}")
    print(f"  use_strict_bf16={use_strict_bf16}")
    print(f"  profile={profile}")
    print(f"  data_source={data_source}")
    print()

    result = train_on_modal.remote(
        batch_size=batch_size,
        max_genes=max_genes,
        n_steps=n_steps,
        learning_rate=learning_rate,
        log_every=log_every,
        model_size=model_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_compile=use_compile,
        compile_mode=compile_mode,
        use_swiglu=use_swiglu,
        use_lp_layernorm=use_lp_layernorm,
        use_softcap=use_softcap,
        use_strict_bf16=use_strict_bf16,
        profile=profile,
        data_source=data_source,
    )

    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    if result.get("status") == "success":
        print("Status: SUCCESS")
        print(f"Steps completed: {result['n_steps']}")
        print(f"Batch size: {result['batch_size']}")
        if result.get("effective_batch_size"):
            print(f"Effective batch size: {result['effective_batch_size']}")
        print(f"Max genes: {result['max_genes']}")
        print(f"Model size: {result['model_size']}")
        print(f"GPU: {result['gpu_name']}")
        print()
        print(f"Total elapsed time: {result.get('elapsed_sec', 0):.1f}s")
        if result.get("training_elapsed_sec") is not None:
            print(f"Training time (compute): {result['training_elapsed_sec']:.1f}s")
        if "mfu_pct" in result:
            print(f"MFU: {result['mfu_pct']}%")
        if "gpu_utilization_pct" in result:
            print(f"GPU utilization (nvidia-smi): {result['gpu_utilization_pct']}%")
        if "sm_efficiency_pct" in result:
            print(f"SM efficiency (dmon): {result['sm_efficiency_pct']}%")
        if "achieved_tflops_total" in result:
            print(f"Achieved TFLOPS (total): {result['achieved_tflops_total']}")
        if "achieved_tflops_per_gpu" in result:
            print(f"Achieved TFLOPS (per GPU): {result['achieved_tflops_per_gpu']}")
        if "throughput_cells_per_sec" in result:
            print(f"Throughput: {result['throughput_cells_per_sec']:.0f} cells/sec")
        if "steps_per_sec" in result:
            print(f"Steps/sec: {result['steps_per_sec']}")
        print()
        print("Performance (excludes first batch warmup):")
        print(f"  Median step time: {result['median_step_time_ms']:.1f} ms")
        print(f"  Avg step time: {result['avg_step_time_ms']:.1f} ms")
        print(f"  Total cells: {result['total_cells']:,}")
        print(f"  Total tokens: {result['total_tokens']:,}")
        print()
        print("Memory:")
        print(f"  Peak memory: {result['peak_memory_gb']:.2f} GB")
        print(f"  Utilization: {result['memory_utilization_pct']:.0f}%")
    else:
        print("Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")

    print("=" * 60)
