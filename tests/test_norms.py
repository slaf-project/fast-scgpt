"""Tests for normalization layers."""

import torch

from fast_scgpt.norms import RMSNorm


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_output_shape(self) -> None:
        """RMSNorm preserves input shape."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_normalized_rms(self) -> None:
        """RMSNorm outputs have approximately unit RMS."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64) * 10  # Large input
        out = norm(x)
        # RMS should be close to 1
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_no_learnable_params(self) -> None:
        """RMSNorm has no learnable parameters."""
        norm = RMSNorm(dim=64)
        assert len(list(norm.parameters())) == 0

    def test_gradient_flow(self) -> None:
        """Gradients flow through RMSNorm."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64, requires_grad=True)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_numerical_stability(self) -> None:
        """RMSNorm handles small inputs without NaN."""
        norm = RMSNorm(dim=64)
        x = torch.randn(2, 10, 64) * 1e-6  # Very small input
        out = norm(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
