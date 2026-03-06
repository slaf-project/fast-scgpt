"""Distributed training for Fast-scGPT using native PyTorch DDP.

Uses queue-based distributed dataloader (slaf DistributedSLAFDataLoader + DistributedDataLoader).
Each rank reads batches from the same Modal Queue; no batch broadcast.

Usage (Modal sets SLAF_QUEUE_NAME before torchrun):
    torchrun --nproc_per_node=8 -m fast_scgpt.train_ddp --slaf_path s3://...
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import modal
import torch
import torch.distributed as dist
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP

from fast_scgpt.config import ModelConfig
from fast_scgpt.model import ScGPT
from fast_scgpt.train import clip_expression_tokens, create_mask

# Indices for timing gather (dl, mask, fwd, bwd, opt, compute_total, total_ms)
_IDX_DL, _IDX_MASK, _IDX_FWD, _IDX_BWD, _IDX_OPT, _IDX_COMPUTE, _IDX_TOTAL = range(7)


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


def _log_timing_all_ranks(
    _rank: int,
    is_main: bool,
    world_size: int,
    log_every: int,
    device: torch.device,
    accum_dl_ms: float,
    accum_mask_ms: float,
    accum_forward_ms: float,
    accum_backward_ms: float,
    accum_optim_ms: float,
) -> None:
    """Gather per-rank timings and log min/max/mean on rank 0."""
    compute_total = (
        accum_mask_ms + accum_forward_ms + accum_backward_ms + accum_optim_ms
    )
    total_ms = accum_dl_ms + compute_total
    local_timing = torch.tensor(
        [
            accum_dl_ms,
            accum_mask_ms,
            accum_forward_ms,
            accum_backward_ms,
            accum_optim_ms,
            compute_total,
            total_ms,
        ],
        dtype=torch.float32,
        device=device,
    )
    tensor_list = [
        torch.zeros(7, dtype=torch.float32, device=device) for _ in range(world_size)
    ]
    dist.all_gather(tensor_list, local_timing)
    stacked = torch.stack(tensor_list)  # (world_size, 7)
    mins = stacked.min(dim=0).values.cpu()
    maxs = stacked.max(dim=0).values.cpu()
    means = stacked.float().mean(dim=0).cpu()
    if is_main:
        logger.info(
            f"  Timing (all {world_size} ranks, last {log_every} step(s)): "
            f"dl min={mins[_IDX_DL]:.0f} max={maxs[_IDX_DL]:.0f} avg={means[_IDX_DL]:.0f}ms | "
            f"mask min={mins[_IDX_MASK]:.0f} max={maxs[_IDX_MASK]:.0f} avg={means[_IDX_MASK]:.0f}ms | "
            f"fwd min={mins[_IDX_FWD]:.0f} max={maxs[_IDX_FWD]:.0f} avg={means[_IDX_FWD]:.0f}ms | "
            f"bwd min={mins[_IDX_BWD]:.0f} max={maxs[_IDX_BWD]:.0f} avg={means[_IDX_BWD]:.0f}ms | "
            f"opt min={mins[_IDX_OPT]:.0f} max={maxs[_IDX_OPT]:.0f} avg={means[_IDX_OPT]:.0f}ms | "
            f"total min={mins[_IDX_TOTAL]:.0f} max={maxs[_IDX_TOTAL]:.0f} avg={means[_IDX_TOTAL]:.0f}ms"
        )


def train_ddp(
    slaf_path: str,
    config: ModelConfig | None = None,
    n_steps: int = 1000,
    batch_size: int = 32,
    max_genes: int = 512,
    learning_rate: float = 1e-4,
    log_every: int = 1,
    gradient_accumulation_steps: int = 1,
    use_gradient_checkpointing: bool = False,
    use_compile: bool = False,
    profile: bool = False,
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

    # Queue-based distributed dataloader: SLAF_QUEUE_NAME must be set by orchestrator (e.g. Modal)
    queue_name = os.environ.get("SLAF_QUEUE_NAME")
    if not queue_name:
        raise RuntimeError(
            "SLAF_QUEUE_NAME not set. Use queue-based DDP from Modal (modal run modal_train_distributed.py)."
        )

    # Rank 0: load SLAF metadata and spawn producer workers (DistributedSLAFDataLoader)
    if is_main:
        try:
            from slaf import SLAFArray
            from slaf.ml.distributed import DistributedSLAFDataLoader
        except ImportError as e:
            raise ImportError(
                "slaf with distributed dataloader required (pip install git+https://github.com/slaf-project/slaf.git@distributed_dataloader)"
            ) from e

        logger.info(f"Loading SLAF from: {slaf_path}")
        t0 = time.time()
        slaf_array = SLAFArray(slaf_path)
        logger.info(f"SLAFArray loaded in {time.time() - t0:.2f}s")

        num_genes = slaf_array.shape[1]
        vocab_size = 4 + num_genes
        # Spawn producers and create queue; do not iterate over this loader
        _producer_loader = DistributedSLAFDataLoader(
            slaf_array=slaf_array,
            tokenizer_type="scgpt",
            batch_size=batch_size,
            max_genes=max_genes,
            vocab_size=vocab_size,
            n_expression_bins=config.n_expression_bins,
            queue_name=queue_name,
            n_workers=8,
            n_scanners=8,
            prefetch_batch_size=16_384,
            prefetch_batch_count=16,
            return_tensors=True,
            queue_timeout=30.0,
            seed=42,
        )
        logger.info(f"DistributedSLAFDataLoader spawned producers, queue={queue_name}")
        # Wait until producers have filled the queue so consumers don't hit empty timeouts
        size = _producer_loader.wait_for_queue(min_batches=50, timeout_seconds=300)
        logger.info(f"Queue has {size} batches, starting consumption")
        # Keep reference so producers stay alive; we consume via DistributedDataLoader below
        _producer_loader_ref = _producer_loader

        vocab_tensor = torch.tensor([num_genes], device=device)
    else:
        vocab_tensor = torch.zeros(1, dtype=torch.long, device=device)

    # Sync so non-main ranks don't attach to the queue before it's ready
    dist.barrier()

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
    if use_compile and device.type == "cuda":
        # DDP optimizer in Dynamo doesn't support higher-order ops (e.g. from
        # autograd through compiled backward). Disable it to avoid NotImplementedError.
        # See https://github.com/pytorch/pytorch/issues/104674
        torch._dynamo.config.optimize_ddp = False
        # Capture .item() and other scalar outputs in the graph to avoid graph breaks
        torch._dynamo.config.capture_scalar_outputs = True
        # Disable Inductor CUDA graphs to avoid OOM (they use large private pools).
        # Saves several GB per GPU; may slightly reduce throughput.
        try:
            if hasattr(torch._inductor.config, "triton") and hasattr(
                torch._inductor.config.triton, "cudagraphs"
            ):
                torch._inductor.config.triton.cudagraphs = False
            elif hasattr(torch._inductor.config, "cudagraphs"):
                torch._inductor.config.cudagraphs = False
        except Exception:
            pass
        if is_main:
            logger.info(
                "Compiling model with torch.compile (mode='reduce-overhead', dynamic=True, cudagraphs=off)..."
            )
        # dynamic=True reduces recompilation when input/mask shapes or sizes vary step-to-step
        base_model = torch.compile(base_model, mode="reduce-overhead", dynamic=True)  # type: ignore[assignment]
        if is_main:
            logger.info("Model compiled successfully")
    model = DDP(base_model, device_ids=[device.index])

    if is_main:
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.1, betas=(0.9, 0.95)
    )

    # All ranks: attach to same queue and create consumer dataloader
    from slaf.distributed.dataloader import DistributedDataLoader

    queue = modal.Queue.from_name(
        queue_name, create_if_missing=False, environment_name="main"
    )
    dataloader = DistributedDataLoader(
        queue,
        batch_size=batch_size,
        return_tensors=True,
        prefetch_factor=2,
        queue_timeout=30.0,
    )
    dataloader_iter = iter(dataloader)

    dist.barrier()

    metrics = DistributedMetrics(world_size=world_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    effective_batch = batch_size * gradient_accumulation_steps * world_size
    if is_main:
        logger.info(
            f"Starting training: {n_steps} steps, batch={batch_size}, "
            f"grad_accum={gradient_accumulation_steps}, effective={effective_batch}"
        )

    step = 0
    micro_step = 0
    accum_step_ms = 0.0
    total_loss = 0.0
    start_time = time.time()
    first_loss = None
    last_loss = None
    # Timing accumulators for profile (dl=queue+transfer, mask, forward, backward, optim)
    accum_dl_ms = 0.0
    accum_mask_ms = 0.0
    accum_forward_ms = 0.0
    accum_backward_ms = 0.0
    accum_optim_ms = 0.0

    optimizer.zero_grad()

    while step < n_steps:
        t_dl_start = time.perf_counter()
        try:
            batch_cpu = next(dataloader_iter)
        except StopIteration:
            logger.info(f"[Rank {rank}] Queue exhausted (StopIteration), exiting loop")
            break
        except Exception as e:
            logger.exception(
                f"[Rank {rank}] Error reading from queue (step={step}): {type(e).__name__} {e}"
            )
            raise

        # Move batch to device (each rank has its own batch from the queue)
        input_ids = batch_cpu["input_ids"].to(device, non_blocking=True)
        attention_mask = batch_cpu["attention_mask"].to(device, non_blocking=True)
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_dl_end = time.perf_counter()
        dl_ms = (t_dl_end - t_dl_start) * 1000

        step_start = time.perf_counter()

        if micro_step == 0:
            optimizer.zero_grad()

        # Forward pass
        model.train()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        input_ids = clip_expression_tokens(
            input_ids, config.vocab_size, config.n_expression_bins
        )

        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        t_after_clip = time.perf_counter()

        masked_input_ids, gene_targets, expr_targets, gene_mask = create_mask(
            input_ids,
            attention_mask,
            mask_token_id=config.mask_token_id,
            gene_token_offset=config.gene_token_offset,
            vocab_size=config.vocab_size,
            expr_token_offset=config.expr_token_offset,
            mask_ratio=0.15,
        )

        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        t_mask = time.perf_counter()

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
            loss = (gene_loss + expr_loss) / gradient_accumulation_steps

        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        t_forward = time.perf_counter()

        # Backward (gradients accumulate across micro-steps)
        loss.backward()

        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        t_backward = time.perf_counter()

        micro_step += 1
        is_accumulation_boundary = micro_step >= gradient_accumulation_steps

        if is_accumulation_boundary:
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_optim = time.perf_counter()
        step_time_ms = (t_optim - step_start) * 1000
        accum_step_ms += step_time_ms

        if profile:
            mask_ms = (t_mask - t_after_clip) * 1000
            forward_ms = (t_forward - t_mask) * 1000
            backward_ms = (t_backward - t_forward) * 1000
            optim_ms = (t_optim - t_backward) * 1000
            accum_dl_ms += dl_ms
            accum_mask_ms += mask_ms
            accum_forward_ms += forward_ms
            accum_backward_ms += backward_ms
            accum_optim_ms += optim_ms

        loss_val = loss.item() * gradient_accumulation_steps  # unscaled for logging
        total_loss += loss_val
        last_loss = loss_val
        if first_loss is None:
            first_loss = loss_val

        if is_accumulation_boundary:
            metrics.update(
                accum_step_ms,
                batch_size * gradient_accumulation_steps,
                device,
            )
            step += 1
            micro_step = 0
            accum_step_ms = 0.0

        if is_main and is_accumulation_boundary and step % log_every == 0:
            avg_loss = total_loss / (log_every * gradient_accumulation_steps)
            logger.info(
                f"Step {step}/{n_steps} | Loss: {avg_loss:.4f} | "
                f"Time: {metrics.step_time_ms:.0f}ms | "
                f"Throughput: {metrics.cells_per_sec:.0f} cells/sec | "
                f"Mem: {metrics.peak_memory_gb:.1f}GB ({metrics.memory_utilization_pct:.0f}%)"
            )
            total_loss = 0.0

        if (
            is_accumulation_boundary
            and step % log_every == 0
            and profile
            and accum_forward_ms > 0
        ):
            _log_timing_all_ranks(
                rank,
                is_main,
                world_size,
                log_every,
                device,
                accum_dl_ms,
                accum_mask_ms,
                accum_forward_ms,
                accum_backward_ms,
                accum_optim_ms,
            )
            accum_dl_ms = accum_mask_ms = accum_forward_ms = accum_backward_ms = (
                accum_optim_ms
            ) = 0.0

    # Summary
    elapsed = time.time() - start_time
    summary = metrics.summary()

    # All-reduce peak memory so we report max across ranks; rank 0 writes metrics for Modal
    peak_gb_tensor = torch.tensor(
        [summary["peak_memory_gb"]], dtype=torch.float32, device=device
    )
    dist.all_reduce(peak_gb_tensor, op=dist.ReduceOp.MAX)
    util_pct_tensor = torch.tensor(
        [summary["memory_utilization_pct"]], dtype=torch.float32, device=device
    )
    dist.all_reduce(util_pct_tensor, op=dist.ReduceOp.MAX)
    # Use sum of step times (actual compute) so Modal reports same throughput as this log
    training_elapsed_sec = sum(metrics._step_times) / 1000.0
    metrics_for_modal = {
        "peak_memory_gb": round(peak_gb_tensor.item(), 2),
        "memory_utilization_pct": round(util_pct_tensor.item(), 1),
        "training_elapsed_sec": round(training_elapsed_sec, 2),
    }
    metrics_file = os.environ.get("FAST_SCGPT_METRICS_FILE")
    if is_main and metrics_file:
        with open(metrics_file, "w") as f:
            json.dump(metrics_for_modal, f, indent=2)

    if is_main:
        # Release producer loader before barrier/cleanup so its teardown (threads, Modal)
        # runs while process group is still alive; avoids SIGABRT on process exit.
        _producer_loader_ref = None
        total_cells = summary["total_cells"] * world_size
        training_time = sum(metrics._step_times) / 1000
        logger.info("─" * 60)
        logger.info(f"Training complete: {step} steps in {elapsed:.1f}s")
        logger.info(f"  World size: {world_size} GPUs")
        logger.info(f"  Median step time: {summary['median_step_time_ms']:.0f}ms")
        if training_time > 0:
            logger.info(f"  Throughput: {total_cells / training_time:.0f} cells/sec")
        else:
            logger.info("  Throughput: N/A (no steps)")
        logger.info(f"  Peak memory: {summary['peak_memory_gb']:.1f}GB")
        if first_loss and last_loss:
            logger.info(f"  Loss: {first_loss:.4f} → {last_loss:.4f}")
        logger.info("─" * 60)

    # Sync all ranks before cleanup (whether we hit n_steps or StopIteration)
    dist.barrier()
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients over N micro-steps before optimizer step (effective_batch = batch_size * this * world_size)",
    )
    parser.add_argument(
        "--model_size", type=str, default="small", choices=["small", "base", "large"]
    )
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile for fused kernels (improves MFU)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Log timing breakdown (dl=queue+transfer, mask, forward, backward, optim)",
    )
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
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_compile=args.use_compile,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
