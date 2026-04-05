"""Modal distributed training script for fast-scGPT on 8x H100.

Uses queue-based distributed dataloader from slaf (via slafdb).
The script deploys the SLAF distributed dataloader Modal app automatically
before training (via slaf.ml.distributed.deploy_dataloader_app), so no separate
deploy step is required.

CPU producer workers (spawned by rank 0 in train_ddp) are stopped explicitly when
training completes so they do not keep running after the benchmark (avoids extra cost).
Queue and Dict are cleaned up in this script's finally block.

Usage:
    # Single node (8x H100)
    modal run modal_train_distributed.py

    # FlashAttention-4 A/B (uses separate Modal image with flash-attn-4)
    modal run modal_train_distributed.py --flash-attn-backend fa4

    # Multi-node (2 nodes = 16x H100, default for --multinode)
    modal run modal_train_distributed.py --multinode

    # Custom configuration
    modal run modal_train_distributed.py --batch-size 64 --n-steps 1000

    # Test with minimal config
    modal run modal_train_distributed.py --batch-size 8 --max-genes 128 --n-steps 50

    # Run in detached mode
    modal run --detach modal_train_distributed.py --n-steps 1000

Results are saved to /data/benchmark_results/ on the volume.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import modal
import modal.experimental

# Create Modal app
app = modal.App("fast-scgpt-distributed")

# Mount existing Modal volume with SLAF datasets
slaf_volume = modal.Volume.from_name("slaf-datasets")

# Build images; libhwloc15, libnl-route-3-200 (and ibverbs) for efa_enabled on multi-node
_distributed_base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "libibverbs-dev",
        "libibverbs1",
        "libhwloc15",
        "libnl-route-3-200",
    )
    .uv_pip_install(
        "torch>=2.4.0,<2.5.0",
        "einops>=0.7",
        "numpy>=1.24",
        "loguru>=0.7",
        "polars>=0.20",
        "pyarrow>=14.0",
        "psutil>=5.9",
        "s3fs>=2024.2",
        "packaging",
        "ninja",
        "modal",
        "slafdb",
    )
)

image_fa3 = _distributed_base.pip_install(
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl",
).add_local_dir("fast_scgpt", "/root/fast_scgpt")

image_fa4 = _distributed_base.uv_pip_install(
    "flash-attn-4>=4.0.0b7,<5",
).add_local_dir("fast_scgpt", "/root/fast_scgpt")


def _run_training(
    *,
    flash_attn_backend: str,
    batch_size: int,
    max_genes: int,
    n_steps: int,
    learning_rate: float,
    log_every: int,
    gradient_accumulation_steps: int,
    model_size: str,
    use_gradient_checkpointing: bool,
    use_compile: bool,
    profile: bool,
    data_source: str,
    cluster_info: Any = None,
) -> dict:
    """Shared training entrypoint for single-node and multi-node.

    When cluster_info is None, runs single-node (this container only).
    When cluster_info is set (from modal.experimental.get_cluster_info()), runs
    as one node in a multi-node job; torchrun args include nnodes/node_rank/master_addr.
    """
    import subprocess
    import sys
    import time

    os.environ["FAST_SCGPT_FLASH_ATTN_BACKEND"] = flash_attn_backend
    sys.path.insert(0, "/root")

    import torch
    from loguru import logger

    is_multinode = cluster_info is not None
    if is_multinode:
        num_nodes = len(cluster_info.container_ips)
        node_rank = cluster_info.rank
        master_addr = cluster_info.container_ips[0]
        num_gpus_total = 8 * num_nodes
        logger.info(
            f"Multi-node: node_rank={node_rank} nnodes={num_nodes} master_addr={master_addr}"
        )
    else:
        num_nodes = 1
        node_rank = 0
        num_gpus_total = torch.cuda.device_count()

    # Log GPU info
    logger.info("=" * 60)
    logger.info(
        "Modal Distributed Training - fast-scGPT"
        + (" (multi-node)" if is_multinode else "")
    )
    logger.info("=" * 60)
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"GPU count (this node): {torch.cuda.device_count()}")

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

    # Verify S3 connectivity (once per node is enough; rank 0 only to avoid log spam)
    if slaf_path.startswith("s3://") and (not is_multinode or node_rank == 0):
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

    # NCCL debugging
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["NCCL_TIMEOUT"] = "300"

    # Queue name for distributed dataloader (all ranks on all nodes attach to same queue).
    # In multi-node, all containers must use the SAME queue name; use cluster_id so node 0 and node 1 agree.
    if is_multinode:
        run_id = cluster_info.cluster_id
    else:
        run_id = os.environ.get("MODAL_REQUEST_ID", str(uuid.uuid4())[:8])
    queue_name = f"fast-scgpt-ddp-{run_id}"
    dict_name = f"{queue_name}-partial-groups"
    os.environ["SLAF_QUEUE_NAME"] = queue_name
    os.environ["SLAF_PARTIAL_GROUPS_DICT"] = (
        dict_name  # same convention as queue so all nodes share one dict
    )
    logger.info(
        f"SLAF_QUEUE_NAME={queue_name}"
        + (f" (cluster_id, node_rank={node_rank})" if is_multinode else "")
    )
    metrics_file_path = "/tmp/fast_scgpt_metrics.json"
    os.environ["FAST_SCGPT_METRICS_FILE"] = (
        metrics_file_path  # train_ddp rank 0 writes peak memory here
    )

    modal.Queue.objects.create(queue_name, allow_existing=True, environment_name="main")
    logger.info("Queue created in workspace (main)")

    num_gpus_per_node = torch.cuda.device_count()
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(num_gpus_per_node),
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
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--model_size",
        model_size,
    ]
    if is_multinode:
        cmd = [
            "torchrun",
            f"--nnodes={num_nodes}",
            f"--node_rank={node_rank}",
            f"--master_addr={master_addr}",
            "--master_port=1234",
            "--nproc_per_node",
            str(num_gpus_per_node),
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
            "--gradient_accumulation_steps",
            str(gradient_accumulation_steps),
            "--model_size",
            model_size,
        ]
    if use_gradient_checkpointing:
        cmd.append("--use_gradient_checkpointing")
    if use_compile:
        cmd.append("--use_compile")
    if profile:
        cmd.append("--profile")

    logger.info("Launching distributed training with command:")
    logger.info(f"  {' '.join(cmd)}")

    start_time = time.time()
    summary = None
    try:
        subprocess.run(
            cmd,
            cwd="/root",
            capture_output=False,
            check=True,
        )
        elapsed = time.time() - start_time
        effective_batch = batch_size * gradient_accumulation_steps * num_gpus_total
        summary = {
            "status": "success",
            "flash_attn_backend": flash_attn_backend,
            "n_steps": n_steps,
            "batch_size_per_gpu": batch_size,
            "effective_batch_size": effective_batch,
            "num_gpus": num_gpus_total,
            "num_nodes": num_nodes,
            "max_genes": max_genes,
            "model_size": model_size,
            "elapsed_sec": elapsed,
            "gpu_name": torch.cuda.get_device_name(0),
        }
        try:
            from fast_scgpt.attention import attention_backend_label

            summary["flash_attn_label"] = attention_backend_label()
        except Exception:
            pass
        # Merge peak memory from train_ddp (rank 0 writes to metrics file)
        try:
            with open(metrics_file_path) as f:
                summary.update(json.load(f))
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Could not read metrics file (non-fatal): {e}")
        # MFU, achieved TFLOPS, throughput, steps/sec
        from fast_scgpt.training_metrics import compute_training_metrics

        summary.update(compute_training_metrics(summary))
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        summary = {"error": f"Training failed: {e}"}
    finally:
        should_cleanup = not is_multinode or node_rank == 0
        if should_cleanup:
            import asyncio

            async def _cleanup() -> None:
                try:
                    await modal.Queue.objects.delete.aio(
                        queue_name, allow_missing=True, environment_name="main"
                    )
                    logger.info(f"Cleaned up queue: {queue_name}")
                except Exception as e:
                    logger.warning(f"Cleanup queue failed (non-fatal): {e}")
                try:
                    await modal.Dict.objects.delete.aio(
                        dict_name, allow_missing=True, environment_name="main"
                    )
                    logger.info(f"Cleaned up dict: {dict_name}")
                except Exception as e:
                    logger.warning(f"Cleanup dict failed (non-fatal): {e}")

            try:
                asyncio.run(_cleanup())
            except Exception as cleanup_err:
                logger.warning(f"Cleanup queue/dict failed (non-fatal): {cleanup_err}")

    if summary is None:
        summary = {"error": "Training did not produce a summary"}

    if summary.get("status") == "success" and should_cleanup:
        from datetime import datetime

        results_dir = "/data/benchmark_results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{results_dir}/distributed_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Results saved to: {results_file}")

    return summary


@app.function(
    image=image_fa3,
    gpu="A100:2",
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
    gradient_accumulation_steps: int = 1,
    model_size: str = "base",
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
    profile: bool = False,
    data_source: str = "s3",
) -> dict:
    """Run distributed training on 8x H100 (single node); flash-attn wheel (FA2/FA3)."""
    return _run_training(
        flash_attn_backend="fa3",
        batch_size=batch_size,
        max_genes=max_genes,
        n_steps=n_steps,
        learning_rate=learning_rate,
        log_every=log_every,
        gradient_accumulation_steps=gradient_accumulation_steps,
        model_size=model_size,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_compile=use_compile,
        profile=profile,
        data_source=data_source,
        cluster_info=None,
    )


@app.function(
    image=image_fa4,
    gpu="A100:2",
    timeout=14400,
    volumes={"/data": slaf_volume},
    secrets=[modal.Secret.from_name("s3-credentials")],
)
def train_distributed_on_modal_fa4(
    batch_size: int = 64,
    max_genes: int = 1024,
    n_steps: int = 500,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    gradient_accumulation_steps: int = 1,
    model_size: str = "base",
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
    profile: bool = False,
    data_source: str = "s3",
) -> dict:
    """Run distributed training on 8x H100 (single node); FlashAttention-4."""
    return _run_training(
        flash_attn_backend="fa4",
        batch_size=batch_size,
        max_genes=max_genes,
        n_steps=n_steps,
        learning_rate=learning_rate,
        log_every=log_every,
        gradient_accumulation_steps=gradient_accumulation_steps,
        model_size=model_size,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_compile=use_compile,
        profile=profile,
        data_source=data_source,
        cluster_info=None,
    )


@app.function(
    image=image_fa3,
    gpu="H100:8",
    timeout=14400,
    volumes={"/data": slaf_volume},
    secrets=[modal.Secret.from_name("s3-credentials")],
    # EFA unlocks extra capacity on H100 multi-node; see modal-labs/multinode-training-guide benchmark
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=2, rdma=True)
def train_distributed_multinode_on_modal(
    batch_size: int = 64,
    max_genes: int = 1024,
    n_steps: int = 500,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    gradient_accumulation_steps: int = 1,
    model_size: str = "base",
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
    profile: bool = False,
    data_source: str = "s3",
) -> dict:
    """Run distributed training on 2 nodes x 8x H100 (16 GPUs). Multi-node (Beta)."""
    cluster_info = modal.experimental.get_cluster_info()
    return _run_training(
        flash_attn_backend="fa3",
        batch_size=batch_size,
        max_genes=max_genes,
        n_steps=n_steps,
        learning_rate=learning_rate,
        log_every=log_every,
        gradient_accumulation_steps=gradient_accumulation_steps,
        model_size=model_size,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_compile=use_compile,
        profile=profile,
        data_source=data_source,
        cluster_info=cluster_info,
    )


@app.function(
    image=image_fa4,
    gpu="H100:8",
    timeout=14400,
    volumes={"/data": slaf_volume},
    secrets=[modal.Secret.from_name("s3-credentials")],
    experimental_options={"efa_enabled": True},
)
@modal.experimental.clustered(size=2, rdma=True)
def train_distributed_multinode_on_modal_fa4(
    batch_size: int = 64,
    max_genes: int = 1024,
    n_steps: int = 500,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    gradient_accumulation_steps: int = 1,
    model_size: str = "base",
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
    profile: bool = False,
    data_source: str = "s3",
) -> dict:
    """Multi-node 16x H100 with FlashAttention-4."""
    cluster_info = modal.experimental.get_cluster_info()
    return _run_training(
        flash_attn_backend="fa4",
        batch_size=batch_size,
        max_genes=max_genes,
        n_steps=n_steps,
        learning_rate=learning_rate,
        log_every=log_every,
        gradient_accumulation_steps=gradient_accumulation_steps,
        model_size=model_size,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_compile=use_compile,
        profile=profile,
        data_source=data_source,
        cluster_info=cluster_info,
    )


@app.local_entrypoint()
def main(
    batch_size: int = 64,
    max_genes: int = 1024,
    n_steps: int = 500,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    gradient_accumulation_steps: int = 1,
    model_size: str = "base",
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
    profile: bool = False,
    data_source: str = "s3",
    multinode: bool = False,
    flash_attn_backend: str = "fa3",
) -> None:
    """Run distributed training benchmark from local machine.

    Ensures the SLAF distributed dataloader Modal app is deployed (via
    deploy_dataloader_app) before launching training, so everything is
    controlled from this repo.

    Args:
        batch_size: Batch size per GPU (default: 64)
        max_genes: Max genes per cell (default: 1024)
        n_steps: Training steps (default: 500)
        learning_rate: LR (default: 1e-4)
        log_every: Log interval (default: 1)
        gradient_accumulation_steps: Accumulate gradients over N micro-steps (default: 1)
        model_size: small/base/large (default: base)
        use_gradient_checkpointing: Trade compute for memory
        use_compile: Use torch.compile for fused kernels
        profile: Log timing breakdown per step
        data_source: Data source - "s3", "volume", or "hf"
        multinode: If True, run on 2 nodes (16x H100). Default False = single node (8x H100).
        flash_attn_backend: "fa3" (flash-attn wheel) or "fa4" (flash-attn-4)
    """
    if flash_attn_backend not in ("fa3", "fa4"):
        raise ValueError("flash_attn_backend must be 'fa3' or 'fa4'")

    if multinode:
        num_gpus = 16  # 2 nodes x 8
        train_fn = (
            train_distributed_multinode_on_modal_fa4
            if flash_attn_backend == "fa4"
            else train_distributed_multinode_on_modal
        )
        mode = "2 nodes (16x H100)"
    else:
        num_gpus = 8
        train_fn = (
            train_distributed_on_modal_fa4
            if flash_attn_backend == "fa4"
            else train_distributed_on_modal
        )
        mode = "8x H100"
    effective_batch = batch_size * gradient_accumulation_steps * num_gpus

    # Deploy SLAF distributed dataloader app so training can spawn dataloader workers.
    try:
        from slaf.ml.distributed import deploy_dataloader_app

        deploy_dataloader_app(show_logs=False, cpu=2, memory=8192)
    except ImportError as e:
        print(
            "ERROR: Could not import deploy_dataloader_app from slaf.ml.distributed. "
            "Ensure slafdb[ml] is installed and up to date (see slaf-project/slaf)."
        )
        raise SystemExit(1) from e

    print(f"Launching fast-scGPT DISTRIBUTED training on Modal ({mode})...")
    print(
        f"  batch_size_per_gpu={batch_size}, grad_accum={gradient_accumulation_steps}, effective_batch={effective_batch}"
    )
    print(f"  max_genes={max_genes}, n_steps={n_steps}")
    print(f"  model_size={model_size}, lr={learning_rate}")
    print(f"  gradient_accumulation_steps={gradient_accumulation_steps}")
    print(f"  use_gradient_checkpointing={use_gradient_checkpointing}")
    print(f"  use_compile={use_compile}")
    print(f"  profile={profile}")
    print(f"  data_source={data_source}")
    print(f"  multinode={multinode}")
    print(f"  flash_attn_backend={flash_attn_backend}")
    print()

    result = train_fn.remote(
        batch_size=batch_size,
        max_genes=max_genes,
        n_steps=n_steps,
        learning_rate=learning_rate,
        log_every=log_every,
        gradient_accumulation_steps=gradient_accumulation_steps,
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
        if result.get("num_nodes"):
            print(f"Nodes: {result['num_nodes']}")
        print(f"Batch size per GPU: {result['batch_size_per_gpu']}")
        print(f"Effective batch size: {result['effective_batch_size']}")
        print(f"Max genes: {result['max_genes']}")
        print(f"Model size: {result['model_size']}")
        print(f"GPU: {result['gpu_name']}")
        print()
        print(f"Total elapsed time: {result['elapsed_sec']:.1f}s")
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
        if "peak_memory_gb" in result:
            print(f"Peak GPU memory: {result['peak_memory_gb']:.2f} GB")
        if "memory_utilization_pct" in result:
            print(f"Memory utilization: {result['memory_utilization_pct']:.1f}%")
    else:
        print("Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")

    print("=" * 60)
