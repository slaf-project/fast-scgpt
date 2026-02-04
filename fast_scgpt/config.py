"""Model configuration for Fast-scGPT."""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for ScGPT model architecture.

    Phase 1 defaults are small (4 layers, 4 heads, 256 dim) for fast iteration.
    Full scGPT uses 12 layers, 8 heads, 512 dim (~53M params).

    Attributes:
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        d_model: Model dimension (embedding size).
        d_ff: Feed-forward hidden dimension. Defaults to 4 * d_model.
        vocab_size: Gene vocabulary size (includes special tokens).
        n_expression_bins: Number of expression level bins for discretization.
        max_seq_len: Maximum sequence length (2 * max_genes + 2 for scGPT format).
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
    """

    # Architecture
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int | None = None  # Defaults to 4 * d_model

    # Vocabulary
    vocab_size: int = 50000
    n_expression_bins: int = 10
    max_seq_len: int = 2050  # 2 * 1024 + 2 (CLS + SEP)

    # Regularization
    dropout: float = 0.1
    bias: bool = False

    # Special token IDs (must match SLAF tokenizer)
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    mask_token_id: int = 3
    gene_token_offset: int = 4  # Genes start at token ID 4

    # Computed fields
    _d_ff: int = field(init=False, repr=False)
    _total_vocab_size: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and compute derived fields."""
        # Compute feed-forward dimension
        self._d_ff = self.d_ff if self.d_ff is not None else 4 * self.d_model

        # Total vocab = genes + special tokens + expression bins
        # Expression bins start at vocab_size
        self._total_vocab_size = self.vocab_size + self.n_expression_bins

        # Validate head dimension
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @property
    def d_head(self) -> int:
        """Per-head dimension."""
        return self.d_model // self.n_heads

    @property
    def ff_dim(self) -> int:
        """Feed-forward dimension (computed or explicit)."""
        return self._d_ff

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary including expression bins."""
        return self._total_vocab_size

    @classmethod
    def small(cls) -> "ModelConfig":
        """Small config for Phase 1 development (4 layers, 256 dim)."""
        return cls(
            n_layers=4,
            n_heads=4,
            d_model=256,
            dropout=0.1,
        )

    @classmethod
    def base(cls) -> "ModelConfig":
        """Base config matching scGPT paper (12 layers, 512 dim, ~53M params)."""
        return cls(
            n_layers=12,
            n_heads=8,
            d_model=512,
            n_expression_bins=51,  # scGPT paper default
            dropout=0.1,
        )

    @classmethod
    def large(cls) -> "ModelConfig":
        """Large config for scaling experiments."""
        return cls(
            n_layers=24,
            n_heads=16,
            d_model=1024,
            n_expression_bins=51,
            dropout=0.1,
        )
