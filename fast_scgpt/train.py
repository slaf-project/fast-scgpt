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
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import torch
from loguru import logger
from torch.autograd.profiler import record_function

from fast_scgpt.config import ModelConfig
from fast_scgpt.device import get_device, get_device_info, get_dtype
from fast_scgpt.gpu_hw_metrics import DmonUtilSampler
from fast_scgpt.model import ScGPT
from fast_scgpt.training_profiler import (
    build_torch_profiler,
    export_chrome_trace,
    format_profiler_report,
    set_torch_profiler_active,
    torch_profiler_active,
)


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
    # nvidia-smi dmon (steady state, after first training step)
    gpu_utilization_pct: float | None = None
    sm_efficiency_pct: float | None = None

    # Cumulative tracking
    _step_times: list[float] = field(default_factory=list)
    _cells_processed: int = 0
    _tokens_processed: int = 0
    _cells_per_optimizer_step: int = 0

    def update(
        self,
        step_time_ms: float,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> None:
        """Update metrics after a training step.

        ``batch_size`` is the number of training examples (cells) per optimizer step
        (``batch_size * gradient_accumulation_steps`` from the training loop).
        """
        self.step_time_ms = step_time_ms
        self._step_times.append(step_time_ms)
        self._cells_per_optimizer_step = batch_size

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

        Avg/median step times exclude the first optimizer step when there are
        multiple steps (compile / cache warmup). ``median_cells_per_sec`` uses
        that median duration (comparable to typical per-step logs).
        """
        # Exclude first batch (warmup/compilation) for robust step-time stats
        times = self._step_times[1:] if len(self._step_times) > 1 else self._step_times

        avg_step_time = sum(times) / len(times) if times else 0
        median_step_time = sorted(times)[len(times) // 2] if times else 0

        median_cells_per_sec = (
            self._cells_per_optimizer_step / (median_step_time / 1000.0)
            if median_step_time > 0 and self._cells_per_optimizer_step > 0
            else 0.0
        )

        out: dict[str, float] = {
            "avg_step_time_ms": avg_step_time,
            "median_step_time_ms": median_step_time,
            "total_cells": self._cells_processed,
            "total_tokens": self._tokens_processed,
            "median_cells_per_sec": median_cells_per_sec,
            "peak_memory_gb": self.peak_memory_gb,
            "memory_utilization_pct": self.memory_utilization_pct,
        }
        if self.gpu_utilization_pct is not None:
            out["gpu_utilization_pct"] = round(self.gpu_utilization_pct, 1)
        if self.sm_efficiency_pct is not None:
            out["sm_efficiency_pct"] = round(self.sm_efficiency_pct, 1)
        return out


def reset_cuda_stats(device: torch.device) -> None:
    """Reset CUDA memory statistics for accurate peak tracking."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()


def create_mask(
    input_ids: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int = 3,
    pad_token_id: int = 0,
    gene_token_offset: int = 4,
    vocab_size: int = 50000,
    expr_token_offset: int = 50000,
    n_expression_bins: int = 200,
    mask_ratio: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random masking for masked gene prediction.

    For canonical dual-stream scGPT format:
    - input_ids: [CLS] gene1 gene2 ... [SEP]
    - values:    [PAD] expr1 expr2 ... [PAD]

    Args:
        input_ids: Token IDs (batch, seq_len)
        values: Expression/value token IDs aligned with input_ids (batch, seq_len)
        attention_mask: Attention mask (batch, seq_len)
        mask_token_id: Token ID for [MASK]
        pad_token_id: Token ID for [PAD]
        gene_token_offset: Offset where gene tokens start
        vocab_size: Size of gene vocabulary (expression bins start after this)
        mask_ratio: Fraction of genes to mask

    Returns:
        Tuple of:
        - masked_input_ids: Gene input with masked tokens replaced
        - masked_values: Value stream with masked positions replaced by PAD
        - gene_targets: Target gene IDs (-100 for non-masked positions)
        - expr_targets: Target expression bins (-100 for non-masked positions)
        - gene_mask: Boolean mask for gene positions that were masked
    """
    if input_ids.shape != values.shape:
        raise ValueError(
            "Dual-stream contract violated: input_ids and values shape mismatch, got "
            f"{tuple(input_ids.shape)} vs {tuple(values.shape)}"
        )
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            "Dual-stream contract violated: input_ids and attention_mask shape mismatch, got "
            f"{tuple(input_ids.shape)} vs {tuple(attention_mask.shape)}"
        )

    # Copy input for masking
    masked_input_ids = input_ids.clone()
    masked_values = values.clone()

    # Initialize targets with -100 (ignore in loss)
    gene_targets = torch.full_like(input_ids, -100)
    expr_targets = torch.full_like(input_ids, -100)

    # Canonical scGPT: genes live directly in input_ids stream.
    is_gene_token = (input_ids >= gene_token_offset) & (input_ids < vocab_size)
    can_mask = is_gene_token & attention_mask

    # Random mask selection
    rand = torch.rand_like(input_ids, dtype=torch.float)
    mask_positions = (rand < mask_ratio) & can_mask

    # Store targets before masking
    gene_targets[mask_positions] = input_ids[mask_positions]

    # Targets come from aligned values stream at the same masked positions.
    expr_bin_ids = values - expr_token_offset
    valid_expr = (expr_bin_ids >= 0) & (expr_bin_ids < n_expression_bins)
    expr_targets[mask_positions & valid_expr] = expr_bin_ids[
        mask_positions & valid_expr
    ]

    # Apply masking to input
    masked_input_ids[mask_positions] = mask_token_id
    # Hide expression signal for masked genes to avoid leakage.
    masked_values[mask_positions] = pad_token_id

    return masked_input_ids, masked_values, gene_targets, expr_targets, mask_positions


def clip_expression_tokens(
    input_ids: torch.Tensor,
    vocab_size: int,
    n_expression_bins: int,
) -> torch.Tensor:
    """Clip expression tokens to valid range.

    Defensive clamp: some tokenizer paths can emit expression bin IDs outside
    the configured range.

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


def _torch_prof_region(name: str) -> AbstractContextManager[Any]:
    return cast(
        AbstractContextManager[Any],
        record_function(name) if torch_profiler_active() else nullcontext(),
    )


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
        scaler: ``torch.amp.GradScaler`` for fp16 autocast only; bf16 uses plain
            ``loss.backward()`` like ``train_ddp``
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

    with _torch_prof_region("scgpt.data"):
        # Move batch to device (non_blocking matches train_ddp; best with pinned CPU tensors)
        _nb = device.type == "cuda"
        input_ids = batch["input_ids"].to(device, non_blocking=_nb)
        values = batch["values"].to(device, non_blocking=_nb)
        attention_mask = batch["attention_mask"].to(device, non_blocking=_nb)

        # Clip value tokens to valid expression range (defensive guard for tokenizer bugs).
        values = clip_expression_tokens(
            values, config.vocab_size, config.n_expression_bins
        )

    if profile and device.type == "cuda":
        torch.cuda.synchronize(device)
        t_data = time.perf_counter()

    with _torch_prof_region("scgpt.mask"):
        # Create masking
        masked_input_ids, masked_values, gene_targets, expr_targets, gene_mask = (
            create_mask(
                input_ids,
                values,
                attention_mask,
                mask_token_id=config.mask_token_id,
                pad_token_id=config.pad_token_id,
                gene_token_offset=config.gene_token_offset,
                vocab_size=config.vocab_size,
                expr_token_offset=config.expr_token_offset,
                n_expression_bins=config.n_expression_bins,
                mask_ratio=0.15,
            )
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

    with _torch_prof_region("scgpt.forward_loss"):
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_autocast
        ):
            loss_dict = model.compute_loss(
                masked_input_ids,
                masked_values,
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
        with _torch_prof_region("scgpt.backward"):
            scaler.scale(loss).backward()
        if is_accumulation_boundary:
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
                t_backward = time.perf_counter()
            with _torch_prof_region("scgpt.optimizer"):
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
        with _torch_prof_region("scgpt.backward"):
            loss.backward()
        if is_accumulation_boundary:
            if profile and device.type == "cuda":
                torch.cuda.synchronize(device)
                t_backward = time.perf_counter()
            with _torch_prof_region("scgpt.optimizer"):
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
    compile_mode: str = "reduce-overhead",
    profile: bool = False,
    use_strict_bf16: bool = False,
    torch_profiler_steps: int = 0,
    torch_profiler_warmup_steps: int = 2,
    torch_profiler_chrome_path: str | None = None,
    torch_profiler_record_shapes: bool = False,
    torch_profiler_with_stack: bool = False,
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
        compile_mode: torch.compile mode: "reduce-overhead" (default, faster compile),
            "max-autotune" (slower compile, often better MFU at large batch), or "default"
        profile: Log timing breakdown (data/mask/forward/backward/optim)
        use_strict_bf16: Keep weights/compute in bf16 without autocast (CUDA bf16 only)
        torch_profiler_steps: If > 0 (CUDA only), run PyTorch profiler for this many
            optimizer steps after ``torch_profiler_warmup_steps``, then log a CUDA/CPU
            summary (and optional Chrome trace). Adds ``scgpt.*`` regions for step
            and submodule breakdown. May graph-break ``torch.compile``; compare with
            compile off for clearer operator names.
        torch_profiler_warmup_steps: Optimizer steps to run before starting capture
            (skip compile / first-batch overhead).
        torch_profiler_chrome_path: If set, write ``export_chrome_trace`` JSON here.
        torch_profiler_record_shapes: Pass through to ``torch.profiler.profile``.
        torch_profiler_with_stack: Pass through to ``torch.profiler.profile`` (heavy).

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
    # Keep config metadata aligned with runtime tokenizer setting.
    config.max_seq_len = max_genes + 2
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

    strict_bf16_active = False
    if use_strict_bf16:
        if device.type != "cuda":
            logger.warning("use_strict_bf16 ignored: not on CUDA")
        elif not torch.cuda.is_bf16_supported():
            logger.warning("use_strict_bf16 ignored: bf16 not supported on this GPU")
        else:
            from fast_scgpt.strict_bf16 import convert_to_strict_bf16

            model = cast(ScGPT, convert_to_strict_bf16(model))
            strict_bf16_active = True
            logger.info(
                "Strict BF16: model in bf16, autocast disabled for training step"
            )

    # Optional: torch.compile for fused kernels and speedup
    if use_compile and device.type == "cuda":
        logger.info("Compiling model with torch.compile (mode={})...", compile_mode)
        model = torch.compile(model, mode=compile_mode)  # type: ignore[assignment]
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
        fused=True,
    )

    # Mixed precision training (CUDA only; disabled when strict bf16 owns dtypes).
    # bf16 autocast: no GradScaler (same as train_ddp). fp16 autocast keeps scaler for stability.
    use_amp = device.type == "cuda" and not strict_bf16_active
    use_bf16_autocast = bool(use_amp and torch.cuda.is_bf16_supported())
    scaler = torch.amp.GradScaler() if use_amp and not use_bf16_autocast else None
    if use_amp:
        amp_dtype = "bf16" if use_bf16_autocast else "fp16"
        if use_bf16_autocast:
            logger.info("Mixed precision enabled: {} (no GradScaler)", amp_dtype)
        else:
            logger.info("Mixed precision enabled: {} with GradScaler", amp_dtype)

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

    # Sequence length (dual stream): CLS + max_genes + SEP
    seq_len = max_genes + 2

    # Initialize GPU metrics tracking
    metrics = GPUMetrics()
    reset_cuda_stats(device)
    hw_sampler: DmonUtilSampler | None = None
    n_cuda_devices = torch.cuda.device_count() if device.type == "cuda" else 0

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

    profiler_ctx: Any = None
    if torch_profiler_steps > 0 and device.type == "cuda":
        need = torch_profiler_warmup_steps + torch_profiler_steps
        if n_steps < need:
            logger.warning(
                "torch_profiler_steps={} with warmup={} needs at least {} optimizer steps; "
                "n_steps={} — profiler may capture fewer steps or end early",
                torch_profiler_steps,
                torch_profiler_warmup_steps,
                need,
                n_steps,
            )

    while step < n_steps:
        if (
            torch_profiler_steps > 0
            and device.type == "cuda"
            and profiler_ctx is None
            and step == torch_profiler_warmup_steps
        ):
            profiler_ctx = build_torch_profiler(
                record_shapes=torch_profiler_record_shapes,
                with_stack=torch_profiler_with_stack,
            )
            profiler_ctx.__enter__()
            set_torch_profiler_active(True)
            logger.info(
                "Torch profiler: capturing {} optimizer steps (after {} warmup steps)",
                torch_profiler_steps,
                torch_profiler_warmup_steps,
            )

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
            # Debug: dual stream values are aligned at identical positions.
            values = batch["values"]
            sample_genes = input_ids[0, 1:11]
            sample_values = values[0, 1:11]
            logger.info(
                "Sample gene tokens (aligned stream): {}", sample_genes.tolist()
            )
            logger.info(
                "Sample value tokens (aligned stream): {}", sample_values.tolist()
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
            if (
                profiler_ctx is not None
                and step == torch_profiler_warmup_steps + torch_profiler_steps
            ):
                set_torch_profiler_active(False)
                profiler_ctx.__exit__(None, None, None)
                logger.info(
                    "Torch profiler summary:\n{}", format_profiler_report(profiler_ctx)
                )
                if torch_profiler_chrome_path:
                    export_chrome_trace(profiler_ctx, torch_profiler_chrome_path)
                    logger.info(
                        "Torch profiler Chrome trace: {}",
                        torch_profiler_chrome_path,
                    )
                profiler_ctx = None

            if (
                device.type == "cuda"
                and n_cuda_devices > 0
                and step == 1
                and hw_sampler is None
            ):
                hw_sampler = DmonUtilSampler(n_gpus=n_cuda_devices)
                if not hw_sampler.start():
                    hw_sampler = None

    if profiler_ctx is not None:
        set_torch_profiler_active(False)
        profiler_ctx.__exit__(None, None, None)
        logger.info(
            "Torch profiler (partial capture before loop exit):\n{}",
            format_profiler_report(profiler_ctx),
        )
        if torch_profiler_chrome_path:
            export_chrome_trace(profiler_ctx, torch_profiler_chrome_path)
            logger.info("Torch profiler Chrome trace: {}", torch_profiler_chrome_path)

    if hw_sampler is not None:
        g_pct, sm_pct = hw_sampler.stop()
        if g_pct is not None:
            metrics.gpu_utilization_pct = g_pct
        if sm_pct is not None:
            metrics.sm_efficiency_pct = sm_pct

    elapsed = time.time() - start_time

    # Log final summary
    summary = metrics.summary()
    logger.info("─" * 60)
    logger.info("Training complete. {} steps in {:.2f}s", step, elapsed)
    logger.info("Summary:")
    logger.info("  Avg step time: {:.1f}ms", summary["avg_step_time_ms"])
    logger.info(
        "  Training throughput: {:.0f} cells/sec",
        summary["median_cells_per_sec"],
    )
    logger.info(
        "  Wall-clock throughput: {:.0f} cells/sec",
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
        if metrics.gpu_utilization_pct is not None:
            logger.info(
                "  GPU utilization (nvidia-smi): {:.1f}%",
                metrics.gpu_utilization_pct,
            )
        if metrics.sm_efficiency_pct is not None:
            logger.info(
                "  SM efficiency (nvidia-smi dmon sm): {:.1f}%",
                metrics.sm_efficiency_pct,
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
    parser.add_argument(
        "--use_strict_bf16",
        action="store_true",
        help="Train with model weights in bf16 and no autocast (CUDA bf16 GPUs only)",
    )
    parser.add_argument(
        "--torch-profiler-steps",
        type=int,
        default=0,
        help="CUDA: capture this many optimizer steps with torch.profiler (0=off)",
    )
    parser.add_argument(
        "--torch-profiler-warmup-steps",
        type=int,
        default=2,
        help="Optimizer steps to skip before torch profiler capture",
    )
    parser.add_argument(
        "--torch-profiler-chrome-path",
        type=str,
        default="",
        help="Write Chrome trace JSON to this path (empty=skip)",
    )
    parser.add_argument(
        "--torch-profiler-record-shapes",
        action="store_true",
        help="Enable profiler record_shapes (heavier)",
    )
    parser.add_argument(
        "--torch-profiler-with-stack",
        action="store_true",
        help="Enable profiler with_stack (heavier)",
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

    chrome = args.torch_profiler_chrome_path or None

    train(
        slaf_path=str(slaf_path),
        config=config,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        max_genes=args.max_genes,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        use_strict_bf16=args.use_strict_bf16,
        torch_profiler_steps=args.torch_profiler_steps,
        torch_profiler_warmup_steps=args.torch_profiler_warmup_steps,
        torch_profiler_chrome_path=chrome,
        torch_profiler_record_shapes=args.torch_profiler_record_shapes,
        torch_profiler_with_stack=args.torch_profiler_with_stack,
    )


if __name__ == "__main__":
    main()
