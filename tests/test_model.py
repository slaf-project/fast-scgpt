"""Tests for Fast-scGPT model."""

import pytest
import torch

from fast_scgpt.config import ModelConfig
from fast_scgpt.device import get_device
from fast_scgpt.model import (
    FeedForward,
    MultiHeadAttention,
    ScGPT,
    TokenEmbedding,
    TransformerBlock,
)


@pytest.fixture
def config() -> ModelConfig:
    """Small config for testing."""
    return ModelConfig.small()


@pytest.fixture
def device() -> torch.device:
    """Get test device."""
    return get_device()


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_small_config(self) -> None:
        """Test small config creation."""
        config = ModelConfig.small()
        assert config.n_layers == 4
        assert config.n_heads == 4
        assert config.d_model == 256

    def test_base_config(self) -> None:
        """Test base config matching scGPT paper."""
        config = ModelConfig.base()
        assert config.n_layers == 12
        assert config.n_heads == 8
        assert config.d_model == 512
        assert config.n_expression_bins == 51

    def test_d_head_computation(self) -> None:
        """Test per-head dimension computation."""
        config = ModelConfig(n_heads=4, d_model=256)
        assert config.d_head == 64

    def test_ff_dim_default(self) -> None:
        """Test feed-forward dimension defaults to 4x model dim."""
        config = ModelConfig(d_model=256)
        assert config.ff_dim == 1024

    def test_ff_dim_explicit(self) -> None:
        """Test explicit feed-forward dimension."""
        config = ModelConfig(d_model=256, d_ff=512)
        assert config.ff_dim == 512

    def test_total_vocab_size(self) -> None:
        """Test total vocab includes expression bins."""
        config = ModelConfig(vocab_size=50000, n_expression_bins=10)
        assert config.total_vocab_size == 50010

    def test_invalid_head_division(self) -> None:
        """Test that invalid head division raises error."""
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(n_heads=3, d_model=256)


class TestTokenEmbedding:
    """Tests for TokenEmbedding."""

    def test_embedding_shape(self, config: ModelConfig, device: torch.device) -> None:
        """Test embedding output shape."""
        emb = TokenEmbedding(config).to(device)
        x = torch.randint(0, config.total_vocab_size, (2, 100), device=device)
        out = emb(x)
        assert out.shape == (2, 100, config.d_model)

    def test_padding_zero(self, config: ModelConfig, device: torch.device) -> None:
        """Test that padding tokens have zero embedding."""
        emb = TokenEmbedding(config).to(device)
        pad_idx = config.pad_token_id
        # Check the raw embedding (before scaling)
        assert torch.allclose(
            emb.embedding.weight[pad_idx], torch.zeros(config.d_model, device=device)
        )


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_attention_shape(self, config: ModelConfig, device: torch.device) -> None:
        """Test attention output shape."""
        attn = MultiHeadAttention(config).to(device)
        x = torch.randn(2, 100, config.d_model, device=device)
        out = attn(x)
        assert out.shape == x.shape

    def test_attention_with_mask(
        self, config: ModelConfig, device: torch.device
    ) -> None:
        """Test attention with padding mask."""
        attn = MultiHeadAttention(config).to(device)
        x = torch.randn(2, 100, config.d_model, device=device)
        mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        mask[:, 50:] = False  # Mask last 50 positions
        out = attn(x, mask)
        assert out.shape == x.shape


class TestFeedForward:
    """Tests for FeedForward."""

    def test_ff_shape(self, config: ModelConfig, device: torch.device) -> None:
        """Test feed-forward output shape."""
        ff = FeedForward(config).to(device)
        x = torch.randn(2, 100, config.d_model, device=device)
        out = ff(x)
        assert out.shape == x.shape


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_block_shape(self, config: ModelConfig, device: torch.device) -> None:
        """Test transformer block output shape."""
        block = TransformerBlock(config).to(device)
        x = torch.randn(2, 100, config.d_model, device=device)
        out = block(x)
        assert out.shape == x.shape

    def test_block_with_mask(self, config: ModelConfig, device: torch.device) -> None:
        """Test transformer block with attention mask."""
        block = TransformerBlock(config).to(device)
        x = torch.randn(2, 100, config.d_model, device=device)
        mask = torch.ones(2, 100, dtype=torch.bool, device=device)
        mask[:, 50:] = False
        out = block(x, mask)
        assert out.shape == x.shape


class TestScGPT:
    """Tests for ScGPT model."""

    def test_model_forward(self, config: ModelConfig, device: torch.device) -> None:
        """Test model forward pass."""
        model = ScGPT(config).to(device)
        batch_size = 2
        seq_len = 100
        input_ids = torch.randint(
            0, config.total_vocab_size, (batch_size, seq_len), device=device
        )
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

        outputs = model(input_ids, attention_mask)

        assert "gene_logits" in outputs
        assert "expr_logits" in outputs
        assert "hidden_states" in outputs
        assert outputs["gene_logits"].shape == (batch_size, seq_len, config.vocab_size)
        assert outputs["expr_logits"].shape == (
            batch_size,
            seq_len,
            config.n_expression_bins,
        )
        assert outputs["hidden_states"].shape == (batch_size, seq_len, config.d_model)

    def test_model_parameters(self, config: ModelConfig) -> None:
        """Test parameter counting."""
        model = ScGPT(config)
        n_params = model.num_parameters
        assert n_params > 0
        # Small model has ~28M params due to large vocab embedding
        assert n_params < 50_000_000

    def test_gradient_flow(self, config: ModelConfig, device: torch.device) -> None:
        """Test that gradients flow through the model."""
        model = ScGPT(config).to(device)
        batch_size = 2
        seq_len = 100
        input_ids = torch.randint(
            0, config.total_vocab_size, (batch_size, seq_len), device=device
        )
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

        outputs = model(input_ids, attention_mask)
        # Use both gene and expr outputs to ensure all heads get gradients
        loss = outputs["gene_logits"].sum() + outputs["expr_logits"].sum()
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_factory_methods(self) -> None:
        """Test factory methods create valid models."""
        small = ScGPT.small()
        assert small.config.n_layers == 4

        base = ScGPT.base()
        assert base.config.n_layers == 12

    def test_from_config(self, config: ModelConfig) -> None:
        """Test creating model from config."""
        model = ScGPT.from_config(config)
        assert model.config == config


class TestTrainMasking:
    """Tests for training masking logic."""

    def test_create_mask_basic(self, config: ModelConfig, device: torch.device) -> None:
        """Test basic masking functionality."""
        from fast_scgpt.train import create_mask

        batch_size = 2
        seq_len = 100
        # Create input in scGPT format: [CLS] gene1 expr1 gene2 expr2 ... [SEP]
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        input_ids[:, 0] = config.cls_token_id
        # Fill gene positions (1, 3, 5, ...) with gene tokens
        for i in range(1, seq_len - 1, 2):
            input_ids[:, i] = config.gene_token_offset + i
        # Fill expression positions (2, 4, 6, ...) with expression tokens
        for i in range(2, seq_len, 2):
            input_ids[:, i] = config.vocab_size + 1  # Expression bin 1
        input_ids[:, -1] = config.sep_token_id

        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

        masked_ids, gene_targets, expr_targets, gene_mask = create_mask(
            input_ids, attention_mask, mask_ratio=0.3
        )

        # Check that some positions were masked
        assert gene_mask.any()
        # Check that masked positions have mask token
        assert (masked_ids[gene_mask] == config.mask_token_id).all()
        # Check that targets are set for masked positions
        assert (gene_targets[gene_mask] != -100).all()
