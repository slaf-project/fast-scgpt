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
# Use CUDA base image for flash-attn compilation
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "torch>=2.0",
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
    # Flash Attention 2 - build from source with CUDA
    .pip_install(
        "flash-attn",
        extra_options="--no-build-isolation",
    )
    # Copy local fast_scgpt package
    .add_local_dir("fast_scgpt", "/root/fast_scgpt")
)


@app.function(
    image=image,
    gpu="A100-80GB",
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
        model_size: Model size preset (small/base/large)
        gradient_accumulation_steps: Accumulate gradients over N micro-batches
            Effective batch = batch_size * gradient_accumulation_steps
        use_gradient_checkpointing: Trade compute for ~50% activation memory savings
        use_compile: Use torch.compile for fused kernels (may speed up training)
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
    elif model_size == "base":
        config = ModelConfig.base()
    elif model_size == "large":
        config = ModelConfig.large()
    else:
        logger.error(f"Unknown model size: {model_size}")
        return {"error": f"Unknown model size: {model_size}"}

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
        profile=profile,
    )

    # Build results dict
    summary = metrics.summary()
    result = {
        "status": "success",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "max_genes": max_genes,
        "model_size": model_size,
        "avg_step_time_ms": summary["avg_step_time_ms"],
        "avg_cells_per_sec": (
            summary["total_cells"] / (summary["avg_step_time_ms"] * n_steps / 1000)
            if summary["avg_step_time_ms"] > 0
            else 0
        ),
        "total_cells": summary["total_cells"],
        "total_tokens": summary["total_tokens"],
        "peak_memory_gb": summary["peak_memory_gb"],
        "memory_utilization_pct": summary["memory_utilization_pct"],
        "gpu_name": torch.cuda.get_device_name(0),
    }

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
        model_size: small/base/large (default: small)
        gradient_accumulation_steps: Effective batch = batch_size * this
        use_gradient_checkpointing: Trade compute for memory
        use_compile: Use torch.compile for fused kernels
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
        print(f"Max genes: {result['max_genes']}")
        print(f"Model size: {result['model_size']}")
        print()
        print("Performance:")
        print(f"  Avg step time: {result['avg_step_time_ms']:.1f} ms")
        print(f"  Avg throughput: {result['avg_cells_per_sec']:.0f} cells/sec")
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
