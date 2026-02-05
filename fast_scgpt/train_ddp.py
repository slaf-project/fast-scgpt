"""Distributed training for Fast-scGPT using native PyTorch DDP.

Simpler than Accelerate - direct control over distributed operations.

Usage:
    torchrun --nproc_per_node=8 -m fast_scgpt.train_ddp --slaf_path s3://...
"""

import argparse
import os
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from fast_scgpt.config import ModelConfig
from fast_scgpt.model import ScGPT
from fast_scgpt.train import clip_expression_tokens, create_mask


@dataclass
class DistributedMetrics:
    """Metrics for distributed training."""

    step_time_ms: float = 0.0
    cells_per_sec: float = 0.0
    peak_memory_gb: float = 0.0
    memory_utilization_pct: float = 0.0
    _step_times: list[float] = field(default_factory=list)
    _cells_processed: int = 0
    world_size: int = 1

    def update(
        self, step_time_ms: float, batch_size: int, device: torch.device
    ) -> None:
        self.step_time_ms = step_time_ms
        self._step_times.append(step_time_ms)
        step_time_sec = step_time_ms / 1000.0
        total_batch = batch_size * self.world_size
        self.cells_per_sec = total_batch / step_time_sec if step_time_sec > 0 else 0
        self._cells_processed += batch_size
        if device.type == "cuda":
            self.peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1e9
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            self.memory_utilization_pct = (self.peak_memory_gb / total_memory) * 100

    def summary(self) -> dict[str, float]:
        times = self._step_times[1:] if len(self._step_times) > 1 else self._step_times
        avg_time = sum(times) / len(times) if times else 0
        median_time = sorted(times)[len(times) // 2] if times else 0
        return {
            "avg_step_time_ms": avg_time,
            "median_step_time_ms": median_time,
            "total_cells": self._cells_processed,
            "peak_memory_gb": self.peak_memory_gb,
            "memory_utilization_pct": self.memory_utilization_pct,
        }


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, world_size, device


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    dist.destroy_process_group()


def broadcast_batch(
    batch_cpu: dict[str, torch.Tensor] | None,
    device: torch.device,
    rank: int,
) -> dict[str, torch.Tensor]:
    """Broadcast batch from rank 0 to all ranks."""
    # First broadcast the shape
    if rank == 0:
        assert batch_cpu is not None
        shape = torch.tensor(batch_cpu["input_ids"].shape, device=device)
    else:
        shape = torch.zeros(2, dtype=torch.long, device=device)

    dist.broadcast(shape, src=0)
    batch_shape = tuple(shape.tolist())

    # Now broadcast the actual tensors
    if rank == 0:
        input_ids = batch_cpu["input_ids"].to(device)  # type: ignore
        attention_mask = batch_cpu["attention_mask"].to(device)  # type: ignore
    else:
        input_ids = torch.zeros(batch_shape, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_shape, dtype=torch.long, device=device)

    dist.broadcast(input_ids, src=0)
    dist.broadcast(attention_mask, src=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def train_ddp(
    slaf_path: str,
    config: ModelConfig | None = None,
    n_steps: int = 1000,
    batch_size: int = 32,
    max_genes: int = 512,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    use_gradient_checkpointing: bool = False,
) -> DistributedMetrics:
    """Train with native PyTorch DDP."""
    # Setup distributed
    rank, world_size, device = setup_distributed()
    is_main = rank == 0

    if config is None:
        config = ModelConfig.small()

    if is_main:
        logger.info("=" * 60)
        logger.info("DDP Training - Fast-scGPT")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Device: {device}")

    # Check Flash Attention
    if is_main:
        from fast_scgpt.attention import check_flash_attn

        check_flash_attn()

    # Only rank 0 loads data
    slaf_array = None
    dataloader = None

    if is_main:
        try:
            from slaf import SLAFArray
            from slaf.ml import SLAFDataLoader
        except ImportError as e:
            raise ImportError("slafdb required") from e

        logger.info(f"Loading SLAF from: {slaf_path}")
        t0 = time.time()
        slaf_array = SLAFArray(slaf_path)
        logger.info(f"SLAFArray loaded in {time.time() - t0:.2f}s")

    # Broadcast vocab info
    if is_main:
        num_genes = slaf_array.shape[1]  # type: ignore
        vocab_tensor = torch.tensor([num_genes], device=device)
    else:
        vocab_tensor = torch.zeros(1, dtype=torch.long, device=device)

    dist.broadcast(vocab_tensor, src=0)
    num_genes = int(vocab_tensor[0].item())
    vocab_size = 4 + num_genes
    config.vocab_size = vocab_size
    config._expr_token_offset = vocab_size
    config._total_vocab_size = vocab_size + config.n_expression_bins

    if is_main:
        logger.info(f"vocab_size={vocab_size}, total={config.total_vocab_size}")

    # Create model on each rank
    base_model = ScGPT(
        config, use_gradient_checkpointing=use_gradient_checkpointing
    ).to(device)
    model = DDP(base_model, device_ids=[device.index])

    if is_main:
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95)
    )

    # Create dataloader on rank 0
    if is_main:
        from slaf.ml import SLAFDataLoader

        dataloader = SLAFDataLoader(
            slaf_array=slaf_array,  # type: ignore
            tokenizer_type="scgpt",
            batch_size=batch_size,
            max_genes=max_genes,
            n_expression_bins=config.n_expression_bins,
            vocab_size=vocab_size,
            use_mixture_of_scanners=True,
            prefetch_batch_size=512000,
            verbose=False,
        )
        logger.info("SLAFDataLoader created")
        dataloader_iter = iter(dataloader)

    # Sync before training
    dist.barrier()

    metrics = DistributedMetrics(world_size=world_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if is_main:
        logger.info(
            f"Starting training: {n_steps} steps, batch={batch_size}, effective={batch_size * world_size}"
        )

    step = 0
    total_loss = 0.0
    start_time = time.time()
    first_loss = None
    last_loss = None

    while step < n_steps:
        # Signal tensor for coordinating termination
        continue_signal = torch.ones(1, dtype=torch.long, device=device)

        # Rank 0 loads batch
        batch_cpu = None
        if is_main:
            logger.debug(f"Step {step}: Loading batch...")
            load_start = time.perf_counter()
            try:
                batch_cpu = next(dataloader_iter)  # type: ignore
                logger.debug(
                    f"Step {step}: Batch loaded in {time.perf_counter() - load_start:.2f}s"
                )
            except StopIteration:
                logger.warning(f"Dataloader exhausted at step {step}")
                continue_signal[0] = 0

        # Sync all ranks before broadcast
        dist.barrier()

        # Broadcast continue signal
        dist.broadcast(continue_signal, src=0)
        if continue_signal[0].item() == 0:
            break

        # Broadcast batch
        batch = broadcast_batch(batch_cpu, device, rank)

        # Timing
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        # Forward pass
        model.train()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        input_ids = clip_expression_tokens(
            input_ids, config.vocab_size, config.n_expression_bins
        )

        masked_input_ids, gene_targets, expr_targets, gene_mask = create_mask(
            input_ids,
            attention_mask,
            mask_token_id=config.mask_token_id,
            gene_token_offset=config.gene_token_offset,
            vocab_size=config.vocab_size,
            expr_token_offset=config.expr_token_offset,
            mask_ratio=0.15,
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(masked_input_ids, attention_mask)
            gene_logits = outputs["gene_logits"]
            gene_loss = torch.nn.functional.cross_entropy(
                gene_logits[gene_mask], gene_targets[gene_mask], ignore_index=-100
            )
            expr_logits = outputs["expr_logits"]
            expr_mask = torch.zeros_like(gene_mask)
            expr_mask[:, 1:] = gene_mask[:, :-1]
            valid_gene_mask = gene_mask.clone()
            valid_gene_mask[:, -1] = False
            expr_loss = torch.nn.functional.cross_entropy(
                expr_logits[expr_mask], expr_targets[valid_gene_mask], ignore_index=-100
            )
            loss = gene_loss + expr_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize(device)
        step_time_ms = (time.perf_counter() - step_start) * 1000

        metrics.update(step_time_ms, batch_size, device)
        loss_val = loss.item()
        total_loss += loss_val
        last_loss = loss_val
        if first_loss is None:
            first_loss = loss_val

        step += 1

        if is_main and step % log_every == 0:
            avg_loss = total_loss / log_every
            logger.info(
                f"Step {step}/{n_steps} | Loss: {avg_loss:.4f} | "
                f"Time: {metrics.step_time_ms:.0f}ms | "
                f"Throughput: {metrics.cells_per_sec:.0f} cells/sec | "
                f"Mem: {metrics.peak_memory_gb:.1f}GB ({metrics.memory_utilization_pct:.0f}%)"
            )
            total_loss = 0.0

    # Summary
    elapsed = time.time() - start_time
    summary = metrics.summary()

    if is_main:
        total_cells = summary["total_cells"] * world_size
        training_time = sum(metrics._step_times) / 1000
        logger.info("─" * 60)
        logger.info(f"Training complete: {step} steps in {elapsed:.1f}s")
        logger.info(f"  World size: {world_size} GPUs")
        logger.info(f"  Median step time: {summary['median_step_time_ms']:.0f}ms")
        logger.info(f"  Throughput: {total_cells / training_time:.0f} cells/sec")
        logger.info(f"  Peak memory: {summary['peak_memory_gb']:.1f}GB")
        if first_loss and last_loss:
            logger.info(f"  Loss: {first_loss:.4f} → {last_loss:.4f}")
        logger.info("─" * 60)

    cleanup_distributed()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="DDP training for Fast-scGPT")
    parser.add_argument("--slaf_path", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_genes", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument(
        "--model_size", type=str, default="small", choices=["small", "base", "large"]
    )
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    if args.model_size == "small":
        config = ModelConfig.small()
    elif args.model_size == "base":
        config = ModelConfig.base()
    else:
        config = ModelConfig.large()

    train_ddp(
        slaf_path=args.slaf_path,
        config=config,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        max_genes=args.max_genes,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )


if __name__ == "__main__":
    main()
