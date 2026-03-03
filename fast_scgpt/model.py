"""Transformer model for scGPT-style single-cell data.

Architecture innovations from modded-nanogpt:
- RMSNorm: Faster normalization without learnable parameters
- Phase 2 will add QK-Norm and ReLU²
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch.utils.checkpoint import checkpoint

from fast_scgpt.config import ModelConfig
from fast_scgpt.lp_layernorm import LPLayerNorm


class TiedLinear(nn.Module):
    """Linear layer that shares weights with an embedding layer.

    Used for weight tying between input embedding and output projection.
    Only uses the first `vocab_size` entries of the embedding.
    """

    def __init__(self, embedding: nn.Embedding, vocab_size: int):
        super().__init__()
        self.embedding = embedding
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply tied linear transformation.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, vocab_size)
        """
        # Use embedding weights as linear layer weights
        # embedding.weight: (total_vocab_size, d_model)
        # We only use [:vocab_size] for gene prediction
        # Linear: y = x @ W.T, where W is (vocab_size, d_model)
        weight = self.embedding.weight[: self.vocab_size]  # (vocab_size, d_model)
        return F.linear(x, weight)  # (batch, seq_len, vocab_size)


class TokenEmbedding(nn.Module):
    """Embedding layer for gene tokens.

    Maps gene token IDs to dense vectors. The vocabulary includes:
    - Special tokens (PAD=0, CLS=1, SEP=2, MASK=3)
    - Gene tokens (starting at offset 4)
    - Expression bin tokens (starting at vocab_size)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=config.total_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_token_id,
        )
        # Scale embeddings by sqrt(d_model) as in Vaswani et al.
        self.scale = math.sqrt(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed tokens.

        Args:
            x: Token IDs of shape (batch, seq_len)

        Returns:
            Embeddings of shape (batch, seq_len, d_model)
        """
        # Check for out-of-bounds tokens (helps debug SLAF vocab mismatches)
        max_token = x.max().item()
        if max_token >= self.embedding.num_embeddings:
            raise ValueError(
                f"Token ID {max_token} exceeds embedding vocab size "
                f"{self.embedding.num_embeddings}. Check config.vocab_size "
                f"matches SLAF tokenizer vocabulary."
            )
        result: torch.Tensor = self.embedding(x) * self.scale
        return result


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with Flash Attention 3 native layout.

    Uses FA3 native (B, T, H, D) layout on H100 for zero transpose overhead.
    Falls back to SDPA on other GPUs.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.dropout_p = config.dropout

        # Q, K, V projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            attention_mask: Boolean mask of shape (batch, seq_len).
                Currently ignored - FA3 doesn't support masks efficiently.

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        from fast_scgpt.attention import attention_native_layout

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to FA3 native layout: (batch, seq_len, n_heads, d_head)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.n_heads)

        # Use FA3 native layout attention (no transpose on H100)
        out = attention_native_layout(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=False,  # Bidirectional attention for scGPT
            scale=self.scale,
        )

        # Reshape back to (batch, seq_len, d_model)
        out = rearrange(out, "b s h d -> b s (h d)")

        result: torch.Tensor = self.out_proj(out)
        return result


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Supports both GELU (default) and SwiGLU (Llama-style) activations.
    SwiGLU: SwiGLU(x, W, V) = Swish(xW) ⊗ xV
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.use_swiglu = config.use_swiglu

        if self.use_swiglu:
            # SwiGLU requires 2 up-projections (gate + value)
            self.gate_proj = nn.Linear(config.d_model, config.ff_dim, bias=config.bias)
            self.up_proj = nn.Linear(config.d_model, config.ff_dim, bias=config.bias)
        else:
            # Standard GELU
            self.fc1 = nn.Linear(config.d_model, config.ff_dim, bias=config.bias)

        self.fc2 = nn.Linear(config.ff_dim, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            x: Input of shape (batch, seq_len, d_model)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        if self.use_swiglu:
            # SwiGLU: Swish(xW) ⊗ xV
            gate = F.silu(self.gate_proj(x))  # silu = swish
            x = gate * self.up_proj(x)
        else:
            # Standard GELU
            x = self.fc1(x)
            x = F.gelu(x)

        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture.

    Supports gradient checkpointing for memory efficiency during training.
    """

    def __init__(self, config: ModelConfig, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ff = FeedForward(config)

        # Choose LayerNorm implementation
        norm_class = LPLayerNorm if config.use_lp_layernorm else nn.LayerNorm
        self.norm1 = norm_class(config.d_model)
        self.norm2 = norm_class(config.d_model)

        self.dropout = nn.Dropout(config.dropout)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Core forward implementation."""
        # Pre-norm attention
        x = x + self.dropout(self.attention(self.norm1(x), attention_mask))
        # Pre-norm feed-forward
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply transformer block with pre-norm residual connections.

        Args:
            x: Input of shape (batch, seq_len, d_model)
            attention_mask: Boolean mask of shape (batch, seq_len)

        Returns:
            Output of shape (batch, seq_len, d_model)
        """
        if self.use_checkpoint and self.training:
            # Gradient checkpointing: recompute forward during backward
            # Trades compute for ~50% memory savings on activations
            result: torch.Tensor = checkpoint(
                self._forward_impl,
                x,
                attention_mask,
                use_reentrant=False,
            )
            return result
        return self._forward_impl(x, attention_mask)


class ScGPT(nn.Module):
    """ScGPT model for single-cell gene expression modeling.

    Architecture follows scGPT paper with interleaved gene-expression format:
    [CLS] gene1 expr1 gene2 expr2 ... [SEP] [PAD]

    The model uses a shared embedding for both gene tokens and expression bins,
    summing them at each position to create the input representation.

    Supports gradient checkpointing for memory-efficient training.
    """

    def __init__(
        self, config: ModelConfig, use_gradient_checkpointing: bool = False
    ) -> None:
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Token embedding (genes + special tokens + expression bins)
        self.embedding = TokenEmbedding(config)

        # Transformer blocks with optional gradient checkpointing
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config, use_checkpoint=use_gradient_checkpointing)
                for _ in range(config.n_layers)
            ]
        )

        # Final layer norm
        norm_class = LPLayerNorm if config.use_lp_layernorm else nn.LayerNorm
        self.norm_f = norm_class(config.d_model)

        # Output heads
        # Gene prediction: predict masked gene tokens (may be replaced by TiedLinear if tie_weights)
        self.gene_head: nn.Linear | TiedLinear = nn.Linear(
            config.d_model, config.vocab_size, bias=config.bias
        )
        # Expression prediction: predict expression bin
        self.expr_head = nn.Linear(
            config.d_model, config.n_expression_bins, bias=config.bias
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Weight tying: Share embedding and gene_head weights
        if config.tie_weights:
            # Replace gene_head with a wrapper that uses embedding weights
            # This avoids creating a separate weight matrix
            self.gene_head = TiedLinear(self.embedding.embedding, config.vocab_size)
            logger.info(
                f"Weight tying enabled: gene_head shares weights with embedding "
                f"(saves {config.vocab_size * config.d_model / 1e6:.1f}M params)"
            )

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with small values for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
                Format: [CLS] gene1 expr1 gene2 expr2 ... [SEP] [PAD]
            attention_mask: Boolean mask of shape (batch, seq_len).
                True for real tokens, False for padding.

        Returns:
            dict with:
                - gene_logits: Logits for gene prediction (batch, seq_len, vocab_size)
                - expr_logits: Logits for expression prediction (batch, seq_len, n_bins)
                - hidden_states: Final hidden states (batch, seq_len, d_model)
        """
        # Embed tokens
        x = self.embedding(input_ids)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer norm
        x = self.norm_f(x)

        # Output heads
        gene_logits = self.gene_head(x)
        expr_logits = self.expr_head(x)

        # Apply logit softcapping if enabled (nanochat optimization)
        if self.config.use_softcap:
            gene_logits = 15.0 * torch.tanh(gene_logits / 15.0)
            expr_logits = 15.0 * torch.tanh(expr_logits / 15.0)

        return {
            "gene_logits": gene_logits,
            "expr_logits": expr_logits,
            "hidden_states": x,
        }

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gene_targets: torch.Tensor,
        expr_targets: torch.Tensor,
        gene_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute masked prediction loss.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            gene_targets: Target gene IDs at masked positions (batch, seq_len)
            expr_targets: Target expression bins at masked positions (batch, seq_len)
            gene_mask: Boolean mask indicating which positions are masked genes
                (not expression bins or special tokens)

        Returns:
            dict with:
                - loss: Total loss (gene_loss + expr_loss)
                - gene_loss: Cross-entropy loss for gene prediction
                - expr_loss: Cross-entropy loss for expression prediction
        """
        outputs = self.forward(input_ids, attention_mask)

        # Gene loss: only on masked gene positions
        gene_logits = outputs["gene_logits"]
        gene_loss = F.cross_entropy(
            gene_logits[gene_mask],
            gene_targets[gene_mask],
            ignore_index=-100,
        )

        # Expression loss: predict expression at expr position (gene_pos + 1)
        # But expr_targets are stored at gene_mask positions for simplicity
        expr_logits = outputs["expr_logits"]

        # Get expression predictions at position gene_pos + 1
        # Shift gene_mask right by 1 to get expression positions
        expr_mask = torch.zeros_like(gene_mask)
        expr_mask[:, 1:] = gene_mask[:, :-1]

        # Also need a gene_mask that excludes genes at the last position
        # (those don't have valid expression positions)
        valid_gene_mask = gene_mask.clone()
        valid_gene_mask[:, -1] = False  # Genes at last position have no expr slot

        # Get targets and logits - targets at gene positions, logits at expr positions
        expr_loss = F.cross_entropy(
            expr_logits[expr_mask],
            expr_targets[valid_gene_mask],
            ignore_index=-100,
        )

        total_loss = gene_loss + expr_loss

        return {
            "loss": total_loss,
            "gene_loss": gene_loss,
            "expr_loss": expr_loss,
        }

    @property
    def num_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Enable or disable gradient checkpointing for all transformer blocks.

        Gradient checkpointing reduces memory usage by ~50% for activations
        by recomputing intermediate values during the backward pass.

        Args:
            enabled: If True, enable checkpointing; if False, disable it.
        """
        self.use_gradient_checkpointing = enabled
        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                block.use_checkpoint = enabled

    @classmethod
    def from_config(cls, config: ModelConfig) -> "ScGPT":
        """Create model from config."""
        return cls(config)

    @classmethod
    def small(cls) -> "ScGPT":
        """Create small model for Phase 1 development."""
        return cls(ModelConfig.small())

    @classmethod
    def base(cls) -> "ScGPT":
        """Create base model matching scGPT paper."""
        return cls(ModelConfig.base())
