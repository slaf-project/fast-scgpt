"""Tests for train_ddp helpers (e.g. producer worker shutdown)."""

from fast_scgpt.train_ddp import _stop_producer_workers


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
