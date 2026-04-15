"""Tests for train_ddp helpers (e.g. producer worker shutdown)."""

import pytest
import torch

from fast_scgpt.train_ddp import (
    _stop_producer_workers,
    _validate_dual_stream_batch_shapes,
)


class TestStopProducerWorkers:
    """Test _stop_producer_workers uses slaf's stop_prefetch_workers when available."""

    def test_calls_stop_prefetch_workers_when_present(self) -> None:
        """When the loader has stop_prefetch_workers(), it is called and no fallback is used."""
        called = []

        class LoaderWithStopPrefetch:
            def stop_prefetch_workers(self) -> None:
                called.append("stop_prefetch_workers")

        loader = LoaderWithStopPrefetch()
        _stop_producer_workers(loader)
        assert called == ["stop_prefetch_workers"]

    def test_ignores_none(self) -> None:
        """None loader is a no-op."""
        _stop_producer_workers(None)  # no raise

    def test_missing_stop_prefetch_swallows_attribute_error(self) -> None:
        """Only stop_prefetch_workers() is used; no shutdown() fallback."""
        called = []

        class LoaderWithShutdownOnly:
            def shutdown(self) -> None:
                called.append("shutdown")

        loader = LoaderWithShutdownOnly()
        _stop_producer_workers(loader)
        assert called == []

    def test_stop_prefetch_exception_logged_but_not_raised(self) -> None:
        """If stop_prefetch_workers() raises, we catch and do not re-raise."""

        class LoaderRaises:
            def stop_prefetch_workers(self) -> None:
                raise RuntimeError("worker stop failed")

        loader = LoaderRaises()
        # Should not raise; exception is logged inside the function
        _stop_producer_workers(loader)


class TestDualStreamBatchContract:
    """Fast fail tests for distributed scGPT batch shape contract."""

    def test_missing_values_raises(self) -> None:
        batch = {
            "input_ids": torch.zeros((2, 8), dtype=torch.long),
            "attention_mask": torch.ones((2, 8), dtype=torch.bool),
        }
        with pytest.raises(ValueError, match="missing required 'values'"):
            _validate_dual_stream_batch_shapes(batch)

    def test_values_input_ids_shape_mismatch_raises(self) -> None:
        batch = {
            "input_ids": torch.zeros((2, 8), dtype=torch.long),
            "values": torch.zeros((2, 7), dtype=torch.long),
            "attention_mask": torch.ones((2, 8), dtype=torch.bool),
        }
        with pytest.raises(ValueError, match="values and input_ids"):
            _validate_dual_stream_batch_shapes(batch)

    def test_values_attention_mask_shape_mismatch_raises(self) -> None:
        batch = {
            "input_ids": torch.zeros((2, 8), dtype=torch.long),
            "values": torch.zeros((2, 8), dtype=torch.long),
            "attention_mask": torch.ones((2, 7), dtype=torch.bool),
        }
        with pytest.raises(ValueError, match="values and attention_mask"):
            _validate_dual_stream_batch_shapes(batch)

    def test_aligned_shapes_pass(self) -> None:
        batch = {
            "input_ids": torch.zeros((2, 8), dtype=torch.long),
            "values": torch.zeros((2, 8), dtype=torch.long),
            "attention_mask": torch.ones((2, 8), dtype=torch.bool),
        }
        _validate_dual_stream_batch_shapes(batch)
