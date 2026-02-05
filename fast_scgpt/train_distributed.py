"""Distributed training for Fast-scGPT using HuggingFace Accelerate.

This module enables multi-GPU training with minimal code changes:
- Data parallel training across multiple GPUs
- Automatic gradient synchronization
- Mixed precision support
- Works on single-node multi-GPU (e.g., 8x H100)

Usage (local):
    accelerate launch --num_processes 8 -m fast_scgpt.train_distributed

Usage (Modal):
    See modal_train_distributed.py
"""

import argparse
import itertools
import time
from dataclasses import dataclass, field

import torch
from accelerate import Accelerator
from loguru import logger

from fast_scgpt.config import ModelConfig
from fast_scgpt.model import ScGPT
from fast_scgpt.train import (
    clip_expression_tokens,
    create_mask,
    reset_cuda_stats,
)


@dataclass
class DistributedMetrics:
    """Metrics for distributed training across multiple GPUs."""

    step_time_ms: float = 0.0
    cells_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    peak_memory_gb: float = 0.0
    memory_utilization_pct: float = 0.0

    # Per-process tracking
    _step_times: list[float] = field(default_factory=list)
    _cells_processed: int = 0
    _tokens_processed: int = 0

    # Global (across all processes)
    world_size: int = 1

    def update(
        self,
        step_time_ms: float,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> None:
        """Update metrics after a training step."""
        self.step_time_ms = step_time_ms
        self._step_times.append(step_time_ms)

        # Throughput (total across all GPUs)
        step_time_sec = step_time_ms / 1000.0
        total_batch = batch_size * self.world_size
        self.cells_per_sec = total_batch / step_time_sec if step_time_sec > 0 else 0
        self.tokens_per_sec = (
            (total_batch * seq_len) / step_time_sec if step_time_sec > 0 else 0
        )

        # Per-process cumulative
        self._cells_processed += batch_size
        self._tokens_processed += batch_size * seq_len

        # GPU memory (local GPU)
        if device.type == "cuda":
            self.peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1e9
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            self.memory_utilization_pct = (self.peak_memory_gb / total_memory) * 100

    def summary(self) -> dict[str, float]:
        """Return summary statistics (local process only).

        Call gather_metrics() to get global statistics across all processes.
        """
        times = self._step_times[1:] if len(self._step_times) > 1 else self._step_times
        avg_step_time = sum(times) / len(times) if times else 0
        median_step_time = sorted(times)[len(times) // 2] if times else 0

        return {
            "avg_step_time_ms": avg_step_time,
            "median_step_time_ms": median_step_time,
            "total_cells": self._cells_processed,
            "total_tokens": self._tokens_processed,
            "peak_memory_gb": self.peak_memory_gb,
            "memory_utilization_pct": self.memory_utilization_pct,
        }


def train_step_distributed(
    model: ScGPT,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    config: ModelConfig,
) -> dict[str, float]:
    """Execute a single distributed training step.

    Args:
        model: The ScGPT model (wrapped by Accelerator)
        batch: Batch from SLAF dataloader with input_ids, attention_mask
        optimizer: The optimizer (wrapped by Accelerator)
        accelerator: HuggingFace Accelerator instance
        config: Model configuration

    Returns:
        dict with loss values
    """
    model.train()

    # Batch is already on correct device via Accelerator
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Clip expression tokens to valid range (workaround for SLAF tokenizer bug)
    input_ids = clip_expression_tokens(
        input_ids, config.vocab_size, config.n_expression_bins
    )

    # Create masking
    masked_input_ids, gene_targets, expr_targets, gene_mask = create_mask(
        input_ids,
        attention_mask,
        mask_token_id=config.mask_token_id,
        gene_token_offset=config.gene_token_offset,
        vocab_size=config.vocab_size,
        expr_token_offset=config.expr_token_offset,
        mask_ratio=0.15,
    )

    # Forward pass with Accelerator's autocast
    with accelerator.autocast():
        loss_dict = model.compute_loss(
            masked_input_ids,
            attention_mask,
            gene_targets,
            expr_targets,
            gene_mask,
        )
        loss = loss_dict["loss"]

    # Backward pass - Accelerator handles gradient sync
    accelerator.backward(loss)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    return {k: v.item() for k, v in loss_dict.items()}


def train_distributed(
    slaf_path: str,
    config: ModelConfig | None = None,
    n_steps: int = 1000,
    batch_size: int = 32,
    max_genes: int = 512,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
) -> DistributedMetrics:
    """Train ScGPT on SLAF data with distributed data parallelism.

    Args:
        slaf_path: Path to SLAF dataset
        config: Model configuration (default: small)
        n_steps: Number of training steps
        batch_size: Batch size per GPU (total = batch_size * num_gpus)
        max_genes: Maximum genes per cell
        learning_rate: Learning rate
        log_every: Log every N steps
        use_gradient_checkpointing: Enable gradient checkpointing
        use_compile: Use torch.compile for fused kernels

    Returns:
        DistributedMetrics with training statistics
    """
    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="bf16")

    # Setup config
    if config is None:
        config = ModelConfig.small()

    # Only log on main process
    is_main = accelerator.is_main_process
    if is_main:
        logger.info("=" * 60)
        logger.info("Distributed Training - Fast-scGPT")
        logger.info("=" * 60)
        logger.info(f"World size: {accelerator.num_processes}")
        logger.info(f"Local rank: {accelerator.local_process_index}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")

    # Check Flash Attention on main process
    if is_main and accelerator.device.type == "cuda":
        from fast_scgpt.attention import check_flash_attn

        check_flash_attn()

    # Load SLAF data
    try:
        from slaf import SLAFArray
        from slaf.ml import SLAFDataLoader
    except ImportError as e:
        logger.error("SLAF not installed. Install with: pip install slafdb")
        raise ImportError("slafdb required for training") from e

    if is_main:
        logger.info(f"Loading SLAF data from: {slaf_path}")

    t0 = time.time()
    slaf_array = SLAFArray(slaf_path)

    if is_main:
        logger.info(f"SLAFArray loaded in {time.time() - t0:.2f}s")

    # Get vocab_size from SLAF metadata
    num_genes = slaf_array.shape[1]
    vocab_size = 4 + num_genes
    config.vocab_size = vocab_size
    config._expr_token_offset = vocab_size
    config._total_vocab_size = vocab_size + config.n_expression_bins

    if is_main:
        logger.info(
            f"SLAF num_genes={num_genes}, vocab_size={vocab_size}, "
            f"total_vocab_size={config.total_vocab_size}"
        )
        logger.info(f"Model config: {config}")

    # Create model
    model = ScGPT(config, use_gradient_checkpointing=use_gradient_checkpointing)

    if is_main:
        logger.info(f"Model parameters: {model.num_parameters:,}")

    # Optional torch.compile
    if use_compile and accelerator.device.type == "cuda":
        if is_main:
            logger.info("Compiling model with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # Create dataloader with distributed sampler
    # SLAFDataLoader handles its own batching, so we wrap it
    t0 = time.time()
    base_dataloader = SLAFDataLoader(
        slaf_array=slaf_array,
        tokenizer_type="scgpt",
        batch_size=batch_size,
        max_genes=max_genes,
        n_expression_bins=config.n_expression_bins,
        vocab_size=vocab_size,
        use_mixture_of_scanners=True,
        prefetch_batch_size=512000,
        verbose=False,
    )

    if is_main:
        logger.info(f"SLAFDataLoader created in {time.time() - t0:.2f}s")

    # Prepare model, optimizer with Accelerator
    # Note: We don't prepare the dataloader since SLAFDataLoader has special iteration
    model, optimizer = accelerator.prepare(model, optimizer)

    # Sequence length
    seq_len = 2 + max_genes * 2

    # Initialize metrics
    metrics = DistributedMetrics(world_size=accelerator.num_processes)
    if accelerator.device.type == "cuda":
        reset_cuda_stats(accelerator.device)

    # Effective batch size across all GPUs
    effective_batch_size = batch_size * accelerator.num_processes

    if is_main:
        logger.info(f"Starting training for {n_steps} steps")
        logger.info(
            f"Batch size per GPU: {batch_size}, Effective batch: {effective_batch_size}"
        )
        logger.info(f"Max genes: {max_genes}, Seq len: {seq_len}")

    # Pre-fetch first batch with retry
    if is_main:
        logger.info("Pre-fetching first batch...")

    max_retries = 3
    first_batch = None
    last_error = None

    for attempt in range(max_retries):
        try:
            dataloader_iter = iter(base_dataloader)
            first_batch = next(dataloader_iter)
            if is_main:
                logger.info(
                    f"First batch received (shape: {first_batch['input_ids'].shape})"
                )
            break
        except Exception as e:
            last_error = str(e)
            if is_main:
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)

    if first_batch is None:
        raise RuntimeError(
            f"Dataloader failed after {max_retries} retries: {last_error}"
        )

    # Move first batch to device
    first_batch = {k: v.to(accelerator.device) for k, v in first_batch.items()}

    # Training loop
    step = 0
    total_loss = 0.0
    start_time = time.time()
    first_loss = None
    last_loss = None

    optimizer.zero_grad()
    batches = itertools.chain([first_batch], dataloader_iter)

    for batch in batches:
        if step >= n_steps:
            break

        # Move batch to device (first batch already moved)
        if step > 0:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}

        # Synchronize for timing
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
        step_start = time.perf_counter()

        # Training step
        loss_dict = train_step_distributed(model, batch, optimizer, accelerator, config)

        # Synchronize after step
        if accelerator.device.type == "cuda":
            torch.cuda.synchronize(accelerator.device)
        step_time_ms = (time.perf_counter() - step_start) * 1000

        # Update metrics
        metrics.update(step_time_ms, batch_size, seq_len, accelerator.device)

        # Track loss
        total_loss += loss_dict["loss"]
        last_loss = loss_dict["loss"]
        if first_loss is None:
            first_loss = loss_dict["loss"]

        step += 1

        # Logging (main process only)
        if is_main and step % log_every == 0:
            avg_loss = total_loss / log_every

            log_parts = [
                f"Step {step}/{n_steps}",
                f"Loss: {avg_loss:.4f}",
                f"Time: {metrics.step_time_ms:.1f}ms/step",
                f"Throughput: {metrics.cells_per_sec:.0f} cells/sec",
            ]

            if accelerator.device.type == "cuda":
                log_parts.append(
                    f"Memory: {metrics.peak_memory_gb:.2f}GB "
                    f"({metrics.memory_utilization_pct:.0f}%)"
                )

            logger.info(" | ".join(log_parts))
            total_loss = 0.0

    # Final summary
    elapsed = time.time() - start_time
    summary = metrics.summary()

    if is_main:
        training_time_sec = sum(metrics._step_times) / 1000
        total_cells = summary["total_cells"] * accelerator.num_processes

        logger.info("─" * 60)
        logger.info(f"Training complete. {step} steps in {elapsed:.2f}s")
        logger.info("Summary:")
        logger.info(f"  World size: {accelerator.num_processes} GPUs")
        logger.info(f"  Avg step time: {summary['avg_step_time_ms']:.1f}ms")
        logger.info(f"  Median step time: {summary['median_step_time_ms']:.1f}ms")
        logger.info(
            f"  Total throughput: {total_cells / training_time_sec:.0f} cells/sec"
        )
        logger.info(f"  Total cells (all GPUs): {total_cells:,}")

        if accelerator.device.type == "cuda":
            logger.info(
                f"  Peak memory (per GPU): {summary['peak_memory_gb']:.2f}GB "
                f"({summary['memory_utilization_pct']:.0f}%)"
            )

        if first_loss is not None and last_loss is not None:
            loss_change = "↓" if last_loss < first_loss else "↑"
            logger.info(f"  Loss: {first_loss:.4f} → {last_loss:.4f} ({loss_change})")

        logger.info("─" * 60)

    return metrics


def main() -> None:
    """Main entry point for distributed training."""
    parser = argparse.ArgumentParser(description="Distributed training for Fast-scGPT")
    parser.add_argument(
        "--slaf_path",
        type=str,
        required=True,
        help="Path to SLAF dataset",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=1000,
        help="Number of training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--max_genes",
        type=int,
        default=512,
        help="Maximum genes per cell",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "base", "large"],
        default="small",
        help="Model size preset",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile",
    )

    args = parser.parse_args()

    # Get config
    if args.model_size == "small":
        config = ModelConfig.small()
    elif args.model_size == "base":
        config = ModelConfig.base()
    else:
        config = ModelConfig.large()

    train_distributed(
        slaf_path=args.slaf_path,
        config=config,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        max_genes=args.max_genes,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_compile=args.use_compile,
    )


if __name__ == "__main__":
    main()
