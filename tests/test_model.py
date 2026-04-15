"""Tests for Fast-scGPT model."""

from dataclasses import replace

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
from fast_scgpt.train import create_mask


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
            0, config.vocab_size, (batch_size, seq_len), device=device
        )
        values = torch.randint(
            config.expr_token_offset,
            config.expr_token_offset + config.n_expression_bins,
            (batch_size, seq_len),
            device=device,
        )
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

        outputs = model(input_ids, values, attention_mask)

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
            0, config.vocab_size, (batch_size, seq_len), device=device
        )
        values = torch.randint(
            config.expr_token_offset,
            config.expr_token_offset + config.n_expression_bins,
            (batch_size, seq_len),
            device=device,
        )
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

        outputs = model(input_ids, values, attention_mask)
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

    def test_sparse_gene_head_matches_dense_loss(
        self, config: ModelConfig, device: torch.device
    ) -> None:
        """Approach (2): masked-only gene projection should match full head + CE."""
        torch.manual_seed(42)
        dense_cfg = replace(config, sparse_gene_head=False)
        sparse_cfg = replace(config, sparse_gene_head=True)
        dense_m = ScGPT(dense_cfg).to(device)
        sparse_m = ScGPT(sparse_cfg).to(device)
        sparse_m.load_state_dict(dense_m.state_dict())
        dense_m.eval()
        sparse_m.eval()

        batch_size, seq_len = 2, 64
        input_ids = torch.randint(
            config.gene_token_offset,
            config.vocab_size,
            (batch_size, seq_len),
            device=device,
        )
        values = torch.randint(
            config.expr_token_offset,
            config.expr_token_offset + config.n_expression_bins,
            (batch_size, seq_len),
            device=device,
        )
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

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

        d = dense_m.compute_loss(
            masked_input_ids,
            masked_values,
            attention_mask,
            gene_targets,
            expr_targets,
            gene_mask,
        )
        s = sparse_m.compute_loss(
            masked_input_ids,
            masked_values,
            attention_mask,
            gene_targets,
            expr_targets,
            gene_mask,
        )

        assert torch.allclose(d["gene_loss"], s["gene_loss"], rtol=0, atol=0)
        assert torch.allclose(d["expr_loss"], s["expr_loss"], rtol=0, atol=0)
        assert torch.allclose(d["loss"], s["loss"], rtol=0, atol=0)

    def test_sparse_gene_head_skip_forward_has_no_gene_logits(
        self, config: ModelConfig, device: torch.device
    ) -> None:
        cfg = replace(config, sparse_gene_head=True)
        model = ScGPT(cfg).to(device)
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device=device
        )
        values = torch.randint(
            config.expr_token_offset,
            config.expr_token_offset + config.n_expression_bins,
            (batch_size, seq_len),
            device=device,
        )
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )
        out = model(input_ids, values, attention_mask, skip_gene_logits=True)
        assert "gene_logits" not in out
        assert out["expr_logits"].shape == (
            batch_size,
            seq_len,
            config.n_expression_bins,
        )


class TestTrainMasking:
    """Tests for training masking logic."""

    def test_create_mask_basic(self, config: ModelConfig, device: torch.device) -> None:
        """Test basic masking functionality."""
        from fast_scgpt.train import create_mask

        batch_size = 2
        seq_len = 100
        # Create dual-stream input: aligned genes + values.
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        values = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        input_ids[:, 0] = config.cls_token_id
        for i in range(1, seq_len - 1):
            input_ids[:, i] = config.gene_token_offset + i
            values[:, i] = config.expr_token_offset + 1  # Expression bin 1
        input_ids[:, -1] = config.sep_token_id

        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

        masked_ids, masked_values, gene_targets, expr_targets, gene_mask = create_mask(
            input_ids,
            values,
            attention_mask,
            mask_token_id=config.mask_token_id,
            pad_token_id=config.pad_token_id,
            gene_token_offset=config.gene_token_offset,
            vocab_size=config.vocab_size,
            expr_token_offset=config.expr_token_offset,
            n_expression_bins=config.n_expression_bins,
            mask_ratio=0.3,
        )

        # Check that some positions were masked
        assert gene_mask.any()
        # Check that masked positions have mask token
        assert (masked_ids[gene_mask] == config.mask_token_id).all()
        # Check masked value positions are hidden
        assert (masked_values[gene_mask] == config.pad_token_id).all()
        # Check that targets are set for masked positions
        assert (gene_targets[gene_mask] != -100).all()
        assert (expr_targets[gene_mask] != -100).all()
