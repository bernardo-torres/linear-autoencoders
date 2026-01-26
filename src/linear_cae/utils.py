import contextlib
import logging
from collections.abc import Mapping
from pathlib import Path

import torch
from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only


class _NoOpEMA:
    """A dummy class to provide a no-op `average_parameters` context manager."""

    @contextlib.contextmanager
    def average_parameters(self):
        """A context manager that does nothing."""
        yield


def get_grad_norm(parameters, norm_type=2.0):
    """
    Calculate the gradient norm of an iterable of parameters.

    Args:
    parameters (Iterable[Tensor]): an iterable of Tensors that will have gradients normalized
    norm_type (float): type of the used p-norm. Can be 'inf' for infinity norm.

    Returns:
    Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0:
        return torch.tensor(0.0)

    first_device = grads[0].device

    norms = []
    for grad in grads:
        norms.append(torch.linalg.vector_norm(grad, norm_type))

    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in norms]), norm_type
    )

    return total_norm


class RankedLogger(logging.LoggerAdapter):
    """A multi-GPU-friendly python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Initializes a multi-GPU-friendly python command line logger that logs on all processes with their rank
        prefixed in the log message.

        :param name: The name of the logger. Default is ``__name__``.
        :param rank_zero_only: Whether to force all logs to only occur on the rank zero process. Default is `False`.
        :param extra: (Optional) A dict-like object which provides contextual information. See `logging.LoggerAdapter`.
        """
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: str, rank: int | None = None, *args, **kwargs) -> None:
        """Delegate a log call to the underlying logger, after prefixing its message with the rank of the process it's
        being logged from. If `'rank'` is provided, then the log will only occur on that rank/process.

        :param level: The level to log at. Look at `logging.__init__.py` for more information.
        :param msg: The message to log.
        :param rank: The rank to log at.
        :param args: Additional args to pass to the underlying logging function.
        :param kwargs: Any additional keyword args to pass to the underlying logging function.
        """
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
            msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    self.logger.log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    self.logger.log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    self.logger.log(level, msg, *args, **kwargs)


log = RankedLogger(__name__, rank_zero_only=True)


def get_resolved_config_path_from_ckpt(ckpt_path: str) -> str:
    """
    Given a checkpoint path, search for the resolved Hydra config file
    in common locations and return its path.

    Args:
        ckpt_path: Path to the checkpoint file.

    Returns:
        The full path to the resolved config file.

    Raises:
        FileNotFoundError: if no config file is found in any of the expected locations.
    """
    ckpt = Path(ckpt_path)
    base_dir = ckpt.parent.parent
    hydra_dir = base_dir / ".hydra"
    checkpoints_dir = base_dir / "checkpoints"

    # List of filenames we'll look for, in order of preference
    filenames = [
        "config_resolved.yaml",
        "resolved_config.yaml",
        "model.yaml",
        "config.yaml",
    ]

    candidates = []

    # 1) In the .hydra directory
    for name in filenames:
        candidates.append(hydra_dir / name)

    # 2) In the .hydra/checkpoints directory
    for name in filenames:
        candidates.append(checkpoints_dir / name)

    # 3) In the base directory
    for name in filenames:
        candidates.append(base_dir / name)

    # Search for the first one that exists
    for path in candidates:
        if path.exists():
            log.info(f"Resolved config path: {path}")
            return str(path)

    # If none found, list all tried locations
    tried = "\n".join(f"  • {p!s}" for p in candidates)
    log.info("Could not find a resolved config file. Looked in:\n" + tried)
    return None
