"""Training loop for Fast-scGPT with SLAF data integration.

Usage:
    python -m fast_scgpt.train --slaf_path path/to/data.slaf

This module implements masked gene expression prediction training:
1. Load batches from SLAFDataLoader (scGPT tokenization)
2. Randomly mask 15-30% of genes
3. Predict both masked gene IDs and expression bins
4. Log loss every N steps
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger

from fast_scgpt.config import ModelConfig
from fast_scgpt.device import get_device, get_device_info, get_dtype
from fast_scgpt.model import ScGPT


@dataclass
class GPUMetrics:
    """GPU metrics collected during training."""

    step_time_ms: float = 0.0
    cells_per_sec: float = 0.0
    tokens_per_sec: float = 0.0
    peak_memory_gb: float = 0.0
    memory_allocated_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    memory_utilization_pct: float = 0.0

    # Cumulative tracking
    _step_times: list[float] = field(default_factory=list)
    _cells_processed: int = 0
    _tokens_processed: int = 0

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

        # Throughput metrics
        step_time_sec = step_time_ms / 1000.0
        self.cells_per_sec = batch_size / step_time_sec if step_time_sec > 0 else 0
        self.tokens_per_sec = (
            (batch_size * seq_len) / step_time_sec if step_time_sec > 0 else 0
        )

        # Cumulative
        self._cells_processed += batch_size
        self._tokens_processed += batch_size * seq_len

        # GPU memory (CUDA only)
        if device.type == "cuda":
            self.peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1e9
            self.memory_allocated_gb = torch.cuda.memory_allocated(device) / 1e9
            self.memory_reserved_gb = torch.cuda.memory_reserved(device) / 1e9

            total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            self.memory_utilization_pct = (self.peak_memory_gb / total_memory) * 100

    def summary(self) -> dict[str, float]:
        """Return summary statistics."""
        avg_step_time = (
            sum(self._step_times) / len(self._step_times) if self._step_times else 0
        )
        return {
            "avg_step_time_ms": avg_step_time,
            "total_cells": self._cells_processed,
            "total_tokens": self._tokens_processed,
            "peak_memory_gb": self.peak_memory_gb,
            "memory_utilization_pct": self.memory_utilization_pct,
        }


def reset_cuda_stats(device: torch.device) -> None:
    """Reset CUDA memory statistics for accurate peak tracking."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def create_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int = 3,
    gene_token_offset: int = 4,
    vocab_size: int = 50000,
    expr_token_offset: int = 50000,
    mask_ratio: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random masking for masked gene prediction.

    For scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]
    We mask at gene positions (odd indices after CLS: 1, 3, 5, ...)
    and also mask the corresponding expression (even indices: 2, 4, 6, ...)

    Args:
        input_ids: Token IDs (batch, seq_len)
        attention_mask: Attention mask (batch, seq_len)
        mask_token_id: Token ID for [MASK]
        gene_token_offset: Offset where gene tokens start
        vocab_size: Size of gene vocabulary (expression bins start after this)
        mask_ratio: Fraction of genes to mask

    Returns:
        Tuple of:
        - masked_input_ids: Input with masked tokens replaced
        - gene_targets: Target gene IDs (-100 for non-masked positions)
        - expr_targets: Target expression bins (-100 for non-masked positions)
        - gene_mask: Boolean mask for gene positions that were masked
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Copy input for masking
    masked_input_ids = input_ids.clone()

    # Initialize targets with -100 (ignore in loss)
    gene_targets = torch.full_like(input_ids, -100)
    expr_targets = torch.full_like(input_ids, -100)

    # Identify gene positions (odd positions after CLS, which is position 0)
    # In scGPT format: pos 1, 3, 5, ... are genes; pos 2, 4, 6, ... are expressions
    position_indices = torch.arange(seq_len, device=device)
    is_gene_position = (position_indices % 2 == 1) & (position_indices > 0)

    # Also need to check that it's a valid gene token (not padding/special)
    is_gene_token = (input_ids >= gene_token_offset) & (input_ids < vocab_size)

    # Combine: gene position AND gene token AND not padding
    can_mask = is_gene_position.unsqueeze(0) & is_gene_token & attention_mask

    # Random mask selection
    rand = torch.rand_like(input_ids, dtype=torch.float)
    mask_positions = (rand < mask_ratio) & can_mask

    # Store targets before masking
    gene_targets[mask_positions] = input_ids[mask_positions]

    # Get corresponding expression targets (position + 1)
    for b in range(batch_size):
        gene_pos = mask_positions[b].nonzero(as_tuple=True)[0]
        expr_pos = gene_pos + 1
        valid_expr_pos = expr_pos < seq_len
        expr_pos = expr_pos[valid_expr_pos]
        gene_pos_valid = gene_pos[valid_expr_pos]

        if len(expr_pos) > 0:
            # Expression tokens are at expr_token_offset + bin_id
            # We need to extract the bin_id as the target
            expr_tokens = input_ids[b, expr_pos]
            expr_targets[b, gene_pos_valid] = expr_tokens - expr_token_offset

    # Apply masking to input
    # For genes: replace with [MASK]
    masked_input_ids[mask_positions] = mask_token_id

    # For expressions: also replace with [MASK] or zero
    # (We'll mask expression positions corresponding to masked genes)
    for b in range(batch_size):
        gene_pos = mask_positions[b].nonzero(as_tuple=True)[0]
        expr_pos = gene_pos + 1
        valid_expr_pos = expr_pos < seq_len
        expr_pos = expr_pos[valid_expr_pos]
        if len(expr_pos) > 0:
            masked_input_ids[b, expr_pos] = mask_token_id

    return masked_input_ids, gene_targets, expr_targets, mask_positions


def train_step(
    model: ScGPT,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
) -> dict[str, float]:
    """Execute a single training step.

    Args:
        model: The ScGPT model
        batch: Batch from SLAF dataloader with input_ids, attention_mask
        optimizer: The optimizer
        config: Model configuration
        device: Device to train on
        scaler: GradScaler for mixed precision (CUDA only)
        use_amp: Whether to use automatic mixed precision

    Returns:
        dict with loss values
    """
    model.train()
    optimizer.zero_grad()

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

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

    # Forward pass with mixed precision
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    use_autocast = use_amp and device.type == "cuda"

    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
        loss_dict = model.compute_loss(
            masked_input_ids,
            attention_mask,
            gene_targets,
            expr_targets,
            gene_mask,
        )
        loss = loss_dict["loss"]

    # Backward pass with gradient scaling
    if scaler is not None and use_autocast:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return {k: v.item() for k, v in loss_dict.items()}


def train(
    slaf_path: str,
    config: ModelConfig | None = None,
    n_steps: int = 1000,
    batch_size: int = 32,
    max_genes: int = 512,
    learning_rate: float = 1e-4,
    log_every: int = 1,
) -> GPUMetrics:
    """Train ScGPT on SLAF data.

    Args:
        slaf_path: Path to SLAF dataset
        config: Model configuration (default: small)
        n_steps: Number of training steps
        batch_size: Batch size
        max_genes: Maximum genes per cell
        learning_rate: Learning rate
        log_every: Log every N steps

    Returns:
        GPUMetrics with training statistics
    """
    # Setup
    if config is None:
        config = ModelConfig.small()

    device = get_device()
    dtype = get_dtype(device)

    logger.info("Device info: {}", get_device_info())
    logger.info("Training dtype: {}", dtype)

    # Load SLAF data FIRST to get vocab_size
    try:
        from slaf import SLAFArray
        from slaf.ml import SLAFDataLoader
    except ImportError as e:
        logger.error("SLAF not installed. Install with: pip install slafdb")
        raise ImportError("slafdb required for training") from e

    logger.info("Loading SLAF data from: {}", slaf_path)
    t0 = time.time()
    slaf_array = SLAFArray(slaf_path)
    logger.info("SLAFArray loaded in {:.2f}s", time.time() - t0)

    # Get vocab_size from SLAF metadata (4 special tokens + num_genes)
    num_genes = slaf_array.shape[1]

    # Update config with actual vocab_size from data
    vocab_size = 4 + num_genes  # 4 special tokens + genes
    config.vocab_size = vocab_size
    # Recompute derived fields
    config._expr_token_offset = vocab_size
    config._total_vocab_size = vocab_size + config.n_expression_bins
    logger.info(
        "SLAF num_genes={}, vocab_size={}, total_vocab_size={}",
        num_genes,
        vocab_size,
        config.total_vocab_size,
    )

    # NOW create model with correct vocab_size
    logger.info("Model config: {}", config)
    model = ScGPT(config).to(device)
    logger.info("Model parameters: {:,}", model.num_parameters)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # Mixed precision training (CUDA only)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        logger.info("Mixed precision enabled: {}", amp_dtype)

    t0 = time.time()
    dataloader = SLAFDataLoader(
        slaf_array=slaf_array,
        tokenizer_type="scgpt",
        batch_size=batch_size,
        max_genes=max_genes,
        n_expression_bins=config.n_expression_bins,
        vocab_size=vocab_size,  # Expression tokens start at vocab_size
        use_mixture_of_scanners=True,
        verbose=False,
    )
    logger.info("SLAFDataLoader created in {:.2f}s", time.time() - t0)

    # Sequence length: CLS + (gene + expr) * max_genes + SEP
    seq_len = 2 + max_genes * 2

    # Initialize GPU metrics tracking
    metrics = GPUMetrics()
    reset_cuda_stats(device)

    # Training loop
    logger.info("Starting training for {} steps", n_steps)
    logger.info(
        "Batch size: {}, Max genes: {}, Seq len: {}", batch_size, max_genes, seq_len
    )
    step = 0
    total_loss = 0.0
    start_time = time.time()
    first_loss = None
    last_loss = None

    logger.info("Fetching first batch from dataloader...")
    t0 = time.time()
    for batch in dataloader:
        if step == 0:
            logger.info("First batch received in {:.2f}s", time.time() - t0)
        if step >= n_steps:
            break

        # Debug: check token bounds on first batch
        if step == 0:
            input_ids = batch["input_ids"]
            max_token = input_ids.max().item()
            min_token = input_ids.min().item()
            logger.info(
                "First batch token stats: min={}, max={}, total_vocab_size={}",
                min_token,
                max_token,
                config.total_vocab_size,
            )
            if max_token >= config.total_vocab_size:
                logger.error(
                    "Token ID {} exceeds vocab size {}! Increase config.vocab_size.",
                    max_token,
                    config.total_vocab_size,
                )
            # Debug: check what's in gene vs expression positions
            # scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]
            # Gene positions: 1, 3, 5, ... (odd after CLS)
            # Expr positions: 2, 4, 6, ... (even after CLS)
            sample = input_ids[0]  # First sample
            gene_positions = sample[1::2][:10]  # First 10 gene tokens
            expr_positions = sample[2::2][:10]  # First 10 expr tokens
            logger.info(
                "Sample gene tokens (pos 1,3,5...): {}", gene_positions.tolist()
            )
            logger.info(
                "Sample expr tokens (pos 2,4,6...): {}", expr_positions.tolist()
            )
            logger.info(
                "Config vocab_size={}, n_expression_bins={}",
                config.vocab_size,
                config.n_expression_bins,
            )

        # Synchronize before timing (CUDA only)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        loss_dict = train_step(model, batch, optimizer, config, device, scaler, use_amp)

        # Synchronize after step for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_end = time.perf_counter()
        step_time_ms = (step_end - step_start) * 1000

        # Update metrics
        metrics.update(step_time_ms, batch_size, seq_len, device)

        total_loss += loss_dict["loss"]
        last_loss = loss_dict["loss"]
        if first_loss is None:
            first_loss = loss_dict["loss"]

        if (step + 1) % log_every == 0:
            avg_loss = total_loss / log_every

            # Build log message with GPU metrics
            log_parts = [
                f"Step {step + 1}/{n_steps}",
                f"Loss: {avg_loss:.4f} (gene: {loss_dict['gene_loss']:.4f}, expr: {loss_dict['expr_loss']:.4f})",
                f"Time: {metrics.step_time_ms:.1f}ms/step",
                f"Throughput: {metrics.cells_per_sec:.0f} cells/sec",
            ]

            # Add memory info for CUDA
            if device.type == "cuda":
                log_parts.append(
                    f"Memory: {metrics.peak_memory_gb:.2f}GB ({metrics.memory_utilization_pct:.0f}%)"
                )

            logger.info(" | ".join(log_parts))
            total_loss = 0.0

        step += 1

    elapsed = time.time() - start_time

    # Log final summary
    summary = metrics.summary()
    logger.info("─" * 60)
    logger.info("Training complete. {} steps in {:.2f}s", step, elapsed)
    logger.info("Summary:")
    logger.info("  Avg step time: {:.1f}ms", summary["avg_step_time_ms"])
    logger.info(
        "  Avg throughput: {:.0f} cells/sec",
        summary["total_cells"] / elapsed if elapsed > 0 else 0,
    )
    logger.info("  Total cells processed: {:,}", summary["total_cells"])
    logger.info("  Total tokens processed: {:,}", summary["total_tokens"])

    if device.type == "cuda":
        logger.info(
            "  Peak memory: {:.2f}GB ({:.0f}% of GPU)",
            summary["peak_memory_gb"],
            summary["memory_utilization_pct"],
        )

    if first_loss is not None and last_loss is not None:
        loss_change = "↓" if last_loss < first_loss else "↑"
        logger.info("  Loss: {:.4f} → {:.4f} ({})", first_loss, last_loss, loss_change)
    logger.info("─" * 60)

    return metrics


def main() -> None:
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train Fast-scGPT on SLAF data")
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
        help="Batch size",
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

    args = parser.parse_args()

    # Validate path
    slaf_path = Path(args.slaf_path)
    if not slaf_path.exists():
        logger.error("SLAF path does not exist: {}", slaf_path)
        sys.exit(1)

    # Get config
    if args.model_size == "small":
        config = ModelConfig.small()
    elif args.model_size == "base":
        config = ModelConfig.base()
    else:
        config = ModelConfig.large()

    train(
        slaf_path=str(slaf_path),
        config=config,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        max_genes=args.max_genes,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
