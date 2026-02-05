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
import itertools
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
        """Return summary statistics.

        Excludes first batch (warmup) and reports median for robustness.
        """
        # Exclude first batch (warmup/compilation)
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

    # Get corresponding expression targets (vectorized)
    # For masked gene at position p, expression is at p+1
    # We need to exclude genes at last position (no room for expression)
    valid_gene_mask = mask_positions.clone()
    valid_gene_mask[:, -1] = False  # Genes at last pos have no expr slot

    # Shift input_ids left by 1 to align expression tokens with gene positions
    # So expr_at_gene_pos[p] = input_ids[p+1]
    expr_at_gene_pos = torch.zeros_like(input_ids)
    expr_at_gene_pos[:, :-1] = input_ids[:, 1:]

    # Extract expression bin targets (token - offset)
    expr_bin_ids = expr_at_gene_pos - expr_token_offset
    expr_targets[valid_gene_mask] = expr_bin_ids[valid_gene_mask]

    # Apply masking to input
    # For genes: replace with [MASK]
    masked_input_ids[mask_positions] = mask_token_id

    # For expressions at gene_pos+1: also replace with [MASK]
    # Shift mask right to get expression positions
    expr_mask_positions = torch.zeros_like(mask_positions)
    expr_mask_positions[:, 1:] = valid_gene_mask[:, :-1]
    masked_input_ids[expr_mask_positions] = mask_token_id

    return masked_input_ids, gene_targets, expr_targets, mask_positions


def clip_expression_tokens(
    input_ids: torch.Tensor,
    vocab_size: int,
    n_expression_bins: int,
) -> torch.Tensor:
    """Clip expression tokens to valid range.

    Workaround for SLAF tokenizer bug where integer expression values
    bypass clipping (see PRDs/BUG-slaf-tokenizer-expression-clipping.md).

    Expression tokens are at positions vocab_size + bin_id.
    This clips bin_id to [0, n_expression_bins - 1].

    Args:
        input_ids: Token IDs (batch, seq_len)
        vocab_size: Size of gene vocabulary (expression tokens start here)
        n_expression_bins: Number of valid expression bins

    Returns:
        Token IDs with expression bins clipped to valid range
    """
    # Identify expression tokens (those >= vocab_size)
    is_expr_token = input_ids >= vocab_size

    if not is_expr_token.any():
        return input_ids

    # Clone to avoid modifying input
    clipped = input_ids.clone()

    # Extract expression bin IDs, clip, and reconstruct tokens
    expr_bins = clipped[is_expr_token] - vocab_size
    clipped_bins = torch.clamp(expr_bins, 0, n_expression_bins - 1)
    clipped[is_expr_token] = clipped_bins + vocab_size

    return clipped


def train_step(
    model: ScGPT,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    is_accumulation_boundary: bool = True,
    profile: bool = False,
) -> dict[str, float]:
    """Execute a single training step with gradient accumulation support.

    Args:
        model: The ScGPT model
        batch: Batch from SLAF dataloader with input_ids, attention_mask
        optimizer: The optimizer
        config: Model configuration
        device: Device to train on
        scaler: GradScaler for mixed precision (CUDA only)
        use_amp: Whether to use automatic mixed precision
        gradient_accumulation_steps: Number of steps to accumulate gradients
        is_accumulation_boundary: If True, perform optimizer step after backward
        profile: If True, return timing breakdown for each phase

    Returns:
        dict with loss values (and timing_* keys if profile=True)
    """
    model.train()

    # Profiling setup
    if profile and device.type == "cuda":
        torch.cuda.synchronize(device)
        t_start = time.perf_counter()

    # Move batch to device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # Clip expression tokens to valid range (workaround for SLAF tokenizer bug)
    input_ids = clip_expression_tokens(
        input_ids, config.vocab_size, config.n_expression_bins
    )

    if profile and device.type == "cuda":
        torch.cuda.synchronize(device)
        t_data = time.perf_counter()

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

    if profile and device.type == "cuda":
        torch.cuda.synchronize(device)
        t_mask = time.perf_counter()

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
        # Scale loss for gradient accumulation
        loss = loss_dict["loss"] / gradient_accumulation_steps

    if profile and device.type == "cuda":
        torch.cuda.synchronize(device)
        t_forward = time.perf_counter()

    # Backward pass with gradient scaling
    if scaler is not None and use_autocast:
        scaler.scale(loss).backward()
        if is_accumulation_boundary:
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
                t_backward = time.perf_counter()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
                t_optim = time.perf_counter()
        else:
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
                t_backward = time.perf_counter()
                t_optim = t_backward  # No optimizer step
    else:
        loss.backward()
        if is_accumulation_boundary:
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
                t_backward = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
                t_optim = time.perf_counter()
        else:
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
                t_backward = time.perf_counter()
                t_optim = t_backward  # No optimizer step

    # Return unscaled loss for logging
    result = {k: v.item() for k, v in loss_dict.items()}

    # Add timing breakdown if profiling
    if profile and device.type == "cuda":
        result["timing_data_ms"] = (t_data - t_start) * 1000
        result["timing_mask_ms"] = (t_mask - t_data) * 1000
        result["timing_forward_ms"] = (t_forward - t_mask) * 1000
        result["timing_backward_ms"] = (t_backward - t_forward) * 1000
        result["timing_optim_ms"] = (t_optim - t_backward) * 1000

    return result


def train(
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
) -> GPUMetrics:
    """Train ScGPT on SLAF data.

    Args:
        slaf_path: Path to SLAF dataset
        config: Model configuration (default: small)
        n_steps: Number of training steps
        batch_size: Batch size (micro-batch if using gradient accumulation)
        max_genes: Maximum genes per cell
        learning_rate: Learning rate
        log_every: Log every N steps
        gradient_accumulation_steps: Accumulate gradients over N micro-batches
            Effective batch size = batch_size * gradient_accumulation_steps
        use_gradient_checkpointing: Enable gradient checkpointing to reduce
            activation memory by ~50% (trades compute for memory)
        use_compile: Use torch.compile for fused kernels and potential speedup
            (may increase compilation time on first step)
        profile: Log timing breakdown (data/mask/forward/backward/optim)

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

    # Check Flash Attention availability
    if device.type == "cuda":
        from fast_scgpt.attention import check_flash_attn

        check_flash_attn()

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
    model = ScGPT(config, use_gradient_checkpointing=use_gradient_checkpointing).to(
        device
    )
    logger.info("Model parameters: {:,}", model.num_parameters)

    # Optional: torch.compile for fused kernels and speedup
    if use_compile and device.type == "cuda":
        logger.info("Compiling model with torch.compile (mode='reduce-overhead')...")
        model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
        logger.info("Model compiled successfully")
    elif use_compile:
        logger.warning("torch.compile requested but not on CUDA - skipping")

    if use_gradient_checkpointing:
        logger.info("Gradient checkpointing enabled (trades compute for memory)")

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
        prefetch_batch_size=512000,
        verbose=False,
    )
    logger.info("SLAFDataLoader created in {:.2f}s", time.time() - t0)

    # Sequence length: CLS + (gene + expr) * max_genes + SEP
    seq_len = 2 + max_genes * 2

    # Initialize GPU metrics tracking
    metrics = GPUMetrics()
    reset_cuda_stats(device)

    # Training loop
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info("Starting training for {} steps", n_steps)
    logger.info(
        "Batch size: {} (effective: {} with {} accumulation steps)",
        batch_size,
        effective_batch_size,
        gradient_accumulation_steps,
    )
    logger.info("Max genes: {}, Seq len: {}", max_genes, seq_len)

    # Pre-fetch first batch with retry logic for intermittent S3 issues
    logger.info("Pre-fetching first batch (with retry for S3 intermittent issues)...")
    max_retries = 3
    first_batch = None
    last_error = None

    for attempt in range(max_retries):
        prefetch_start = time.time()
        try:
            # Create fresh iterator on each retry
            dataloader_iter = iter(dataloader)
            first_batch = next(dataloader_iter)
            logger.info(
                "First batch received in {:.2f}s (shape: {}, attempt {}/{})",
                time.time() - prefetch_start,
                first_batch["input_ids"].shape,
                attempt + 1,
                max_retries,
            )
            break  # Success
        except StopIteration:
            last_error = "Dataloader returned no batches"
            logger.warning(
                "Attempt {}/{}: {} after {:.1f}s",
                attempt + 1,
                max_retries,
                last_error,
                time.time() - prefetch_start,
            )
        except Exception as e:
            last_error = str(e)
            logger.warning(
                "Attempt {}/{}: Failed after {:.1f}s: {}",
                attempt + 1,
                max_retries,
                time.time() - prefetch_start,
                e,
            )

        if attempt < max_retries - 1:
            wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
            logger.info("Retrying in {}s...", wait_time)
            time.sleep(wait_time)

    if first_batch is None:
        logger.error("Failed to fetch first batch after {} attempts", max_retries)
        raise RuntimeError(
            f"Dataloader failed after {max_retries} retries: {last_error}"
        )

    step = 0
    micro_step = 0  # Tracks position within gradient accumulation window
    total_loss = 0.0
    start_time = time.time()
    first_loss = None
    last_loss = None

    # Accumulated timing across micro-steps (reset after each optimizer step)
    accum_dl_ms = 0.0
    accum_data_ms = 0.0
    accum_mask_ms = 0.0
    accum_fwd_ms = 0.0
    accum_bwd_ms = 0.0
    accum_optim_ms = 0.0
    accum_step_ms = 0.0

    # Initialize gradients
    optimizer.zero_grad()

    # Process pre-fetched batch first, then continue with iterator
    batches = itertools.chain([first_batch], dataloader_iter)
    batch_iter = iter(batches)
    dataloader_time_ms = 0.0  # Track time waiting for dataloader

    while step < n_steps:
        # Measure dataloader wait time
        dl_start = time.perf_counter()
        try:
            batch = next(batch_iter)
        except StopIteration:
            logger.warning("Dataloader exhausted before completing {} steps", n_steps)
            break
        dl_end = time.perf_counter()
        dataloader_time_ms = (dl_end - dl_start) * 1000

        # Debug: check token bounds on first batch
        if step == 0 and micro_step == 0:
            input_ids = batch["input_ids"]
            max_token = input_ids.max().item()
            min_token = input_ids.min().item()
            logger.info(
                "First batch token stats: min={}, max={}, total_vocab_size={}",
                min_token,
                max_token,
                config.total_vocab_size,
            )
            # Check if clipping will be needed (before clipping is applied)
            max_expr_bin = (
                max_token - config.vocab_size if max_token >= config.vocab_size else 0
            )
            if max_expr_bin >= config.n_expression_bins:
                logger.warning(
                    "Expression bin {} exceeds n_expression_bins={}. "
                    "Tokens will be clipped (SLAF tokenizer bug workaround).",
                    max_expr_bin,
                    config.n_expression_bins,
                )
            if max_token >= config.total_vocab_size:
                logger.warning(
                    "Token ID {} exceeds vocab size {}. Will be clipped to valid range.",
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

        # Determine if this is the last micro-batch in accumulation window
        micro_step += 1
        is_accumulation_boundary = micro_step >= gradient_accumulation_steps

        loss_dict = train_step(
            model,
            batch,
            optimizer,
            config,
            device,
            scaler,
            use_amp,
            gradient_accumulation_steps,
            is_accumulation_boundary,
            profile=profile,
        )

        # Synchronize after step for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_end = time.perf_counter()
        step_time_ms = (step_end - step_start) * 1000

        # Accumulate timing across micro-steps
        accum_dl_ms += dataloader_time_ms
        accum_step_ms += step_time_ms
        if profile and "timing_data_ms" in loss_dict:
            accum_data_ms += loss_dict["timing_data_ms"]
            accum_mask_ms += loss_dict["timing_mask_ms"]
            accum_fwd_ms += loss_dict["timing_forward_ms"]
            accum_bwd_ms += loss_dict["timing_backward_ms"]
            accum_optim_ms += loss_dict["timing_optim_ms"]

        # Accumulate loss (train_step returns unscaled loss)
        total_loss += loss_dict["loss"]
        last_loss = loss_dict["loss"]
        if first_loss is None:
            first_loss = loss_dict["loss"]

        # Only count as a "step" when we've done optimizer.step()
        if is_accumulation_boundary:
            micro_step = 0  # Reset for next accumulation window

            # Update metrics with TOTAL time for all micro-steps
            metrics.update(accum_step_ms, effective_batch_size, seq_len, device)

            if (step + 1) % log_every == 0:
                avg_loss = total_loss / (log_every * gradient_accumulation_steps)

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

                # Log timing breakdown if profiling (accumulated across micro-steps)
                if profile and accum_fwd_ms > 0:
                    compute_total = (
                        accum_data_ms
                        + accum_mask_ms
                        + accum_fwd_ms
                        + accum_bwd_ms
                        + accum_optim_ms
                    )
                    logger.info(
                        "  Timing ({}x accum): dl={:.0f}ms | data={:.0f}ms mask={:.0f}ms fwd={:.0f}ms bwd={:.0f}ms opt={:.0f}ms | total={:.0f}ms",
                        gradient_accumulation_steps,
                        accum_dl_ms,
                        accum_data_ms,
                        accum_mask_ms,
                        accum_fwd_ms,
                        accum_bwd_ms,
                        accum_optim_ms,
                        accum_dl_ms + compute_total,
                    )

                total_loss = 0.0

            # Reset accumulators for next optimizer step
            accum_dl_ms = 0.0
            accum_data_ms = 0.0
            accum_mask_ms = 0.0
            accum_fwd_ms = 0.0
            accum_bwd_ms = 0.0
            accum_optim_ms = 0.0
            accum_step_ms = 0.0

            step += 1

    elapsed = time.time() - start_time
    training_time_sec = sum(metrics._step_times) / 1000  # Convert ms to sec

    # Log final summary
    summary = metrics.summary()
    logger.info("─" * 60)
    logger.info("Training complete. {} steps in {:.2f}s", step, elapsed)
    logger.info("Summary:")
    logger.info("  Avg step time: {:.1f}ms", summary["avg_step_time_ms"])
    logger.info(
        "  Training throughput: {:.0f} cells/sec (excludes startup)",
        summary["total_cells"] / training_time_sec if training_time_sec > 0 else 0,
    )
    logger.info(
        "  Wall-clock throughput: {:.0f} cells/sec (includes startup)",
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
