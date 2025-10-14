import contextlib


class _NoOpEMA:
    """A dummy class to provide a no-op `average_parameters` context manager."""

    @contextlib.contextmanager
    def average_parameters(self):
        """A context manager that does nothing."""
        yield
