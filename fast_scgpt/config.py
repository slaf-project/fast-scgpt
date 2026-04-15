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
        max_seq_len: Maximum sequence length (max_genes + 2 for dual-stream scGPT).
        dropout: Dropout probability.
        bias: Whether to use bias in linear layers.
    """

    # Architecture
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int | None = None  # Defaults to 4 * d_model

    # Vocabulary
    # vocab_size = special tokens (4) + num_genes
    # For Tahoe100M: 4 + 62710 = 62714
    # expr_token_offset should equal vocab_size so expressions don't collide with genes
    # Adding buffer for safety (SLAF may have additional tokens)
    vocab_size: int = 62714  # 4 special + 62710 genes
    n_expression_bins: int = 200  # Extra headroom (SLAF uses up to ~150)
    max_seq_len: int = 1026  # 1024 + 2 (CLS + SEP)

    # Regularization
    dropout: float = 0.1
    bias: bool = False

    # Weight tying
    tie_weights: bool = False  # Share embedding and gene_head weights

    # Feed-forward activation
    use_swiglu: bool = False  # Use SwiGLU instead of GELU (Llama-style)

    # Low-precision LayerNorm
    use_lp_layernorm: bool = False  # Force LayerNorm to stay in bf16 (Tahoe-X1 style)

    # Logit softcapping (nanochat optimization)
    use_softcap: bool = False  # Apply tanh softcapping to prevent extreme logits

    # Gene head: if True, training runs F.linear only at masked gene positions (no full
    # (batch, seq_len, vocab_size) logits tensor). Inference / forward(skip_gene_logits=False)
    # unchanged. See PRD 013 / approach (2).
    sparse_gene_head: bool = False

    # Special token IDs (must match SLAF tokenizer)
    pad_token_id: int = 0
    cls_token_id: int = 1
    sep_token_id: int = 2
    mask_token_id: int = 3
    gene_token_offset: int = 4  # Genes start at token ID 4
    # expr_token_offset is computed in __post_init__ to equal vocab_size

    # Computed fields
    _d_ff: int = field(init=False, repr=False)
    _total_vocab_size: int = field(init=False, repr=False)
    _expr_token_offset: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and compute derived fields."""
        # Compute feed-forward dimension
        self._d_ff = self.d_ff if self.d_ff is not None else 4 * self.d_model

        # Expression tokens start right after gene tokens (no overlap)
        self._expr_token_offset = self.vocab_size

        # Total vocab = vocab_size + expression bins
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

    @property
    def expr_token_offset(self) -> int:
        """Offset where expression tokens start (equals vocab_size)."""
        return self._expr_token_offset

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
        """Base config matching scGPT paper (12 layers, 512 dim, ~100M params).

        Note: This uses 8 heads and 4× FF expansion (standard transformer).
        For the actual scGPT architecture (51M params), use scgpt_matched() instead.
        """
        return cls(
            n_layers=12,
            n_heads=8,
            d_model=512,
            n_expression_bins=51,  # scGPT paper default
            dropout=0.1,
        )

    @classmethod
    def scgpt_matched(cls) -> "ModelConfig":
        """Config matching actual scGPT implementation (51M params).

        Verified from scGPT source code (Tutorial_Annotation.ipynb):
        - 12 layers, 4 heads (not 8!), 512 dim
        - 1× FF expansion (d_hid = d_model = 512, not 4×)
        - Weight tying between embedding and gene_head
        - ~51M parameters (2× smaller than base config)

        This is the efficient baseline used in the original scGPT paper.
        """
        return cls(
            n_layers=12,
            n_heads=4,  # Actual scGPT uses 4, not 8
            d_model=512,
            d_ff=512,  # 1× expansion, not 4×!
            n_expression_bins=51,
            dropout=0.2,  # scGPT uses 0.2
            bias=False,
            tie_weights=True,  # Share embedding and gene_head weights
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
