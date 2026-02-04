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
from pathlib import Path

import torch
from loguru import logger

from fast_scgpt.config import ModelConfig
from fast_scgpt.device import get_device, get_device_info, get_dtype
from fast_scgpt.model import ScGPT


def create_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_token_id: int = 3,
    gene_token_offset: int = 4,
    vocab_size: int = 50000,
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
            # Expression tokens are at vocab_size + bin_id
            # We need to extract the bin_id as the target
            expr_tokens = input_ids[b, expr_pos]
            expr_targets[b, gene_pos_valid] = expr_tokens - vocab_size

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
) -> dict[str, float]:
    """Execute a single training step.

    Args:
        model: The ScGPT model
        batch: Batch from SLAF dataloader with input_ids, attention_mask
        optimizer: The optimizer
        config: Model configuration
        device: Device to train on

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
        mask_ratio=0.15,
    )

    # Forward pass with masking
    loss_dict = model.compute_loss(
        masked_input_ids,
        attention_mask,
        gene_targets,
        expr_targets,
        gene_mask,
    )

    # Backward pass
    loss = loss_dict["loss"]
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
    log_every: int = 10,
) -> None:
    """Train ScGPT on SLAF data.

    Args:
        slaf_path: Path to SLAF dataset
        config: Model configuration (default: small)
        n_steps: Number of training steps
        batch_size: Batch size
        max_genes: Maximum genes per cell
        learning_rate: Learning rate
        log_every: Log every N steps
    """
    # Setup
    if config is None:
        config = ModelConfig.small()

    device = get_device()
    dtype = get_dtype(device)

    logger.info("Device info: {}", get_device_info())
    logger.info("Model config: {}", config)
    logger.info("Training dtype: {}", dtype)

    # Create model
    model = ScGPT(config).to(device)
    logger.info("Model parameters: {:,}", model.num_parameters)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # Load SLAF data
    try:
        from slaf import SLAFArray
        from slaf.ml import SLAFDataLoader
    except ImportError as e:
        logger.error("SLAF not installed. Install with: pip install slafdb")
        raise ImportError("slafdb required for training") from e

    logger.info("Loading SLAF data from: {}", slaf_path)
    slaf_array = SLAFArray(slaf_path)

    dataloader = SLAFDataLoader(
        slaf_array=slaf_array,
        tokenizer_type="scgpt",
        batch_size=batch_size,
        max_genes=max_genes,
        n_expression_bins=config.n_expression_bins,
        use_mixture_of_scanners=True,
        verbose=False,
    )

    # Training loop
    logger.info("Starting training for {} steps", n_steps)
    step = 0
    total_loss = 0.0
    start_time = time.time()

    for batch in dataloader:
        if step >= n_steps:
            break

        loss_dict = train_step(model, batch, optimizer, config, device)
        total_loss += loss_dict["loss"]

        if (step + 1) % log_every == 0:
            avg_loss = total_loss / log_every
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed

            logger.info(
                "Step {}/{} | Loss: {:.4f} (gene: {:.4f}, expr: {:.4f}) | "
                "{:.2f} steps/sec",
                step + 1,
                n_steps,
                avg_loss,
                loss_dict["gene_loss"],
                loss_dict["expr_loss"],
                steps_per_sec,
            )
            total_loss = 0.0

        step += 1

    elapsed = time.time() - start_time
    logger.info("Training complete. {} steps in {:.2f}s", step, elapsed)


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
