from typing import Union

import logging
import os
import sys

import numpy as np


__all__ = ["__version__", "logger", "config"]


# read library version from metadata
try:
    import importlib.metadata

    __version__ = importlib.metadata.version("skrl")
except ImportError:
    __version__ = "unknown"


# logger with format
class _Formatter(logging.Formatter):
    _format = "[%(name)s:%(levelname)s] %(message)s"
    _formats = {
        logging.DEBUG: f"\x1b[38;20m{_format}\x1b[0m",
        logging.INFO: f"\x1b[38;20m{_format}\x1b[0m",
        logging.WARNING: f"\x1b[33;20m{_format}\x1b[0m",
        logging.ERROR: f"\x1b[31;20m{_format}\x1b[0m",
        logging.CRITICAL: f"\x1b[31;1m{_format}\x1b[0m",
    }

    def format(self, record):
        return logging.Formatter(self._formats.get(record.levelno)).format(record)


_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(_Formatter())

logger = logging.getLogger("skrl")
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)


# machine learning framework configuration
class _Config(object):
    def __init__(self) -> None:
        """Machine learning framework specific configuration."""

        class PyTorch(object):
            def __init__(self) -> None:
                """PyTorch configuration."""
                # torch.distributed config
                self._local_rank = int(os.getenv("LOCAL_RANK", "0"))
                self._rank = int(os.getenv("RANK", "0"))
                self._world_size = int(os.getenv("WORLD_SIZE", "1"))
                self._is_distributed = self._world_size > 1
                # device
                self._device = f"cuda:{self._local_rank}"

                # set up distributed runs
                if self._is_distributed:
                    import torch

                    logger.info(
                        f"Distributed (rank: {self._rank}, local rank: {self._local_rank}, world size: {self._world_size})"
                    )
                    torch.distributed.init_process_group("nccl", rank=self._rank, world_size=self._world_size)
                    torch.cuda.set_device(self._local_rank)

            @staticmethod
            def parse_device(device: Union[str, "torch.device", None], validate: bool = True) -> "torch.device":
                """Parse the input device and return a :py:class:`~torch.device` instance.

                :param device: Device specification. If the specified device is ``None`` or it cannot be resolved,
                               the default available device will be returned instead.
                :param validate: Whether to check that the specified device is valid. Since PyTorch does not check if
                                 the specified device index is valid, a tensor is created for the verification.

                :return: PyTorch device.
                """
                import torch

                _device = None
                if isinstance(device, torch.device):
                    _device = device
                elif isinstance(device, str):
                    try:
                        _device = torch.device(device)
                    except RuntimeError as e:
                        logger.warning(f"Invalid device specification ({device}): {e}")
                if _device is None:
                    _device = torch.device(
                        "cuda:0" if torch.cuda.is_available() else "cpu"
                    )  # torch.get_default_device() was introduced in version 2.3.0
                # validate device
                if validate:
                    try:
                        torch.zeros((1,), device=_device)
                    except Exception as e:
                        logger.warning(f"Invalid device specification ({device}): {e}")
                        _device = PyTorch.parse_device(None)
                return _device

            @property
            def device(self) -> "torch.device":
                """Default device.

                The default device, unless specified, is ``cuda:0`` (or ``cuda:LOCAL_RANK`` in a distributed environment)
                if CUDA is available, ``cpu`` otherwise.
                """
                self._device = self.parse_device(self._device, validate=False)
                return self._device

            @device.setter
            def device(self, device: Union[str, "torch.device"]) -> None:
                self._device = device

            @property
            def local_rank(self) -> int:
                """The rank of the worker/process (e.g.: GPU) within a local worker group (e.g.: node).

                This property reads from the ``LOCAL_RANK`` environment variable (``0`` if it doesn't exist).

                Read-only attribute.
                """
                return self._local_rank

            @property
            def rank(self) -> int:
                """The rank of the worker/process (e.g.: GPU) within a worker group (e.g.: across all nodes).

                This property reads from the ``RANK`` environment variable (``0`` if it doesn't exist).

                Read-only attribute.
                """
                return self._rank

            @property
            def world_size(self) -> int:
                """The total number of workers/process (e.g.: GPUs) in a worker group (e.g.: across all nodes).

                This property reads from the ``WORLD_SIZE`` environment variable (``1`` if it doesn't exist).

                Read-only attribute.
                """
                return self._world_size

            @property
            def is_distributed(self) -> bool:
                """Whether if running in a distributed environment.

                This property is ``True`` when the PyTorch's distributed environment variable ``WORLD_SIZE > 1``.

                Read-only attribute.
                """
                return self._is_distributed

        self.torch = PyTorch()


config = _Config()
