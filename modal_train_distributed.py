"""Modal distributed training script for fast-scGPT on 8x H100.

This script deploys distributed training to Modal's multi-GPU infrastructure.

Usage:
    # Run with defaults (8x H100, 500 steps)
    modal run modal_train_distributed.py

    # Custom configuration
    modal run modal_train_distributed.py --batch-size 64 --n-steps 1000

    # Test with minimal config
    modal run modal_train_distributed.py --batch-size 8 --max-genes 128 --n-steps 50

    # Run in detached mode
    modal run --detach modal_train_distributed.py --n-steps 1000

Results are saved to /data/benchmark_results/ on the volume.
"""

import modal

# Create Modal app
app = modal.App("fast-scgpt-distributed")

# Mount existing Modal volume with SLAF datasets
slaf_volume = modal.Volume.from_name("slaf-datasets")

# Build image with CUDA + dependencies + Accelerate
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
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
        "accelerate>=0.27",  # HuggingFace Accelerate for distributed
    )
    # Flash Attention - FA3 on H100
    .pip_install(
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
    )
    # Copy local fast_scgpt package
    .add_local_dir("fast_scgpt", "/root/fast_scgpt")
)


@app.function(
    image=image,
    gpu="H100:8",  # 8x H100 in single node
    timeout=14400,  # 4 hours max
    volumes={"/data": slaf_volume},
    secrets=[modal.Secret.from_name("s3-credentials")],
)
def train_distributed_on_modal(
    batch_size: int = 64,
    max_genes: int = 1024,
    n_steps: int = 500,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    model_size: str = "base",
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
    profile: bool = False,
    data_source: str = "s3",
) -> dict:
    """Run distributed training on 8x H100 Modal.

    Args:
        batch_size: Batch size per GPU (total = batch_size * 8)
        max_genes: Maximum genes per cell (affects seq_len)
        n_steps: Number of training steps
        learning_rate: Learning rate
        log_every: Log interval
        model_size: Model size preset (small/base/large)
        use_gradient_checkpointing: Trade compute for ~50% activation memory savings
        use_compile: Use torch.compile for fused kernels
        profile: Log timing breakdown (data/mask/forward/backward/optim)
        data_source: Data source - "s3", "volume", or "hf"

    Returns:
        dict with training summary metrics
    """
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/root")

    import torch
    from loguru import logger

    # Log GPU info
    logger.info("=" * 60)
    logger.info("Modal Distributed Training - fast-scGPT")
    logger.info("=" * 60)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory: {props.total_memory / 1e9:.1f} GB")
            logger.info(f"  Compute: {props.major}.{props.minor}")
    else:
        logger.error("CUDA not available!")
        return {"error": "CUDA not available"}

    # Select data source
    if data_source == "volume":
        slaf_path = "/data/tigris/Tahoe100M_train_SLAF"
    elif data_source == "hf":
        slaf_path = "hf://datasets/slaf-project/Tahoe-100M/data/train"
    elif data_source == "s3":
        slaf_path = "s3://slaf-datasets/Tahoe100M_train_SLAF"
    else:
        logger.error(f"Unknown data_source: {data_source}")
        return {"error": f"Unknown data_source: {data_source}"}

    logger.info(f"Data source: {data_source}")
    logger.info(f"SLAF path: {slaf_path}")

    # Verify S3 connectivity
    if slaf_path.startswith("s3://"):
        logger.info("Verifying S3 connectivity...")
        try:
            import s3fs

            fs = s3fs.S3FileSystem()
            bucket_path = slaf_path.replace("s3://", "")
            files = fs.ls(bucket_path)
            logger.info(f"S3 path accessible, found {len(files)} items")
        except Exception as e:
            logger.error(f"S3 connectivity check failed: {e}")
            return {"error": f"S3 access failed: {e}"}

    # Memory optimization
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Build torchrun command for native DDP
    num_gpus = torch.cuda.device_count()
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(num_gpus),
        "-m",
        "fast_scgpt.train_ddp",
        "--slaf_path",
        slaf_path,
        "--n_steps",
        str(n_steps),
        "--batch_size",
        str(batch_size),
        "--max_genes",
        str(max_genes),
        "--learning_rate",
        str(learning_rate),
        "--log_every",
        str(log_every),
        "--model_size",
        model_size,
    ]

    if use_gradient_checkpointing:
        cmd.append("--use_gradient_checkpointing")
    # Note: use_compile and profile not yet implemented in train_ddp.py
    _ = use_compile, profile  # Silence unused warnings

    logger.info("Launching distributed training with command:")
    logger.info(f"  {' '.join(cmd)}")

    # Run distributed training
    start_time = __import__("time").time()

    try:
        subprocess.run(
            cmd,
            cwd="/root",
            capture_output=False,  # Stream output directly
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        return {"error": f"Training failed: {e}"}

    elapsed = __import__("time").time() - start_time

    # Build results
    effective_batch = batch_size * num_gpus
    summary = {
        "status": "success",
        "n_steps": n_steps,
        "batch_size_per_gpu": batch_size,
        "effective_batch_size": effective_batch,
        "num_gpus": num_gpus,
        "max_genes": max_genes,
        "model_size": model_size,
        "elapsed_sec": elapsed,
        "gpu_name": torch.cuda.get_device_name(0),
    }

    # Save results to volume
    import json
    from datetime import datetime

    results_dir = "/data/benchmark_results"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{results_dir}/distributed_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    return summary


@app.local_entrypoint()
def main(
    batch_size: int = 64,
    max_genes: int = 1024,
    n_steps: int = 500,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    model_size: str = "base",
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
    profile: bool = False,
    data_source: str = "s3",
) -> None:
    """Run distributed training benchmark from local machine.

    Args:
        batch_size: Batch size per GPU (default: 64)
        max_genes: Max genes per cell (default: 1024)
        n_steps: Training steps (default: 500)
        learning_rate: LR (default: 1e-4)
        log_every: Log interval (default: 1)
        model_size: small/base/large (default: base)
        use_gradient_checkpointing: Trade compute for memory
        use_compile: Use torch.compile for fused kernels
        profile: Log timing breakdown per step
        data_source: Data source - "s3", "volume", or "hf"
    """
    # Estimate effective batch size
    num_gpus = 8  # H100:8 configuration
    effective_batch = batch_size * num_gpus

    print("Launching fast-scGPT DISTRIBUTED training on Modal (8x H100)...")
    print(f"  batch_size_per_gpu={batch_size}, effective_batch={effective_batch}")
    print(f"  max_genes={max_genes}, n_steps={n_steps}")
    print(f"  model_size={model_size}, lr={learning_rate}")
    print(f"  use_gradient_checkpointing={use_gradient_checkpointing}")
    print(f"  use_compile={use_compile}")
    print(f"  profile={profile}")
    print(f"  data_source={data_source}")
    print()

    result = train_distributed_on_modal.remote(
        batch_size=batch_size,
        max_genes=max_genes,
        n_steps=n_steps,
        learning_rate=learning_rate,
        log_every=log_every,
        model_size=model_size,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_compile=use_compile,
        profile=profile,
        data_source=data_source,
    )

    print()
    print("=" * 60)
    print("DISTRIBUTED BENCHMARK RESULTS")
    print("=" * 60)

    if result.get("status") == "success":
        print("Status: SUCCESS")
        print(f"Steps completed: {result['n_steps']}")
        print(f"GPUs used: {result['num_gpus']}")
        print(f"Batch size per GPU: {result['batch_size_per_gpu']}")
        print(f"Effective batch size: {result['effective_batch_size']}")
        print(f"Max genes: {result['max_genes']}")
        print(f"Model size: {result['model_size']}")
        print(f"GPU: {result['gpu_name']}")
        print()
        print(f"Total elapsed time: {result['elapsed_sec']:.1f}s")
    else:
        print("Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")

    print("=" * 60)
