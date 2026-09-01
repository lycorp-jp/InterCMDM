"""General runtime helpers shared by training and evaluation entry points."""

from __future__ import annotations

import importlib
import random
import time
from argparse import Namespace
from collections.abc import Mapping, Sequence

import numpy as np
import torch


def _format_duration(seconds: float) -> str:
    whole_seconds = max(0, int(seconds))
    minutes, remainder = divmod(whole_seconds, 60)
    return f"{minutes}m {remainder}s"


def _progress_timing(start_time: float, completed: int, total: int):
    if total <= 0:
        raise ValueError("total_niters must be positive")
    fraction = completed / total
    elapsed = max(0.0, time.time() - start_time)
    if fraction <= 0:
        remaining = 0.0
    else:
        remaining = max(0.0, elapsed * (1.0 - fraction) / fraction)
    timing = f"{_format_duration(elapsed)} (- {_format_duration(remaining)})"
    return fraction, timing


def print_current_loss(
    start_time,
    niter_state,
    total_niters,
    losses,
    epoch=None,
    sub_epoch=None,
    inner_iter=None,
    tf_ratio=None,
    sl_steps=None,
):
    """Print one compact progress line for the current training window."""
    del sub_epoch, tf_ratio, sl_steps
    fraction, timing = _progress_timing(start_time, niter_state, total_niters)
    fields = []
    if epoch is not None:
        displayed_inner_iter = -1 if inner_iter is None else inner_iter
        fields.append(
            f"ep/it:{epoch:2d}-{displayed_inner_iter:4d} niter:{niter_state:6d}"
        )
    fields.append(f"{timing} completed:{fraction * 100:3.0f}%")
    fields.extend(f"{name}: {value:.5f}" for name, value in losses.items())
    print(" ".join(fields))


def fixseed(seed):
    """Seed Python, NumPy, and PyTorch and disable cuDNN autotuning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _serialize_config_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Namespace):
        return namespace_to_dict(value)
    if isinstance(value, Mapping):
        return {key: _serialize_config_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize_config_value(item) for item in value]
    return str(value)


def namespace_to_dict(namespace):
    """Convert a namespace or mapping into values accepted by loggers."""
    if isinstance(namespace, Namespace):
        source = vars(namespace)
    elif isinstance(namespace, Mapping):
        source = namespace
    else:
        raise TypeError(
            "namespace_to_dict expects argparse.Namespace or Mapping, "
            f"received {type(namespace)!r}"
        )
    return {key: _serialize_config_value(value) for key, value in source.items()}


class WandbLogger:
    """Minimal scalar-logger adapter that becomes a no-op without wandb."""

    def __init__(self, args=None):
        self._wandb = None
        self._run = None
        try:
            wandb = importlib.import_module("wandb")
        except ImportError:
            return

        self._wandb = wandb
        run = wandb.run
        if run is None and args is not None:
            run = wandb.init(
                name=getattr(args, "name", None),
                project=args.project_name,
                entity=args.entity,
                config=namespace_to_dict(args),
            )
        self._run = run

    def add_scalar(self, tag, value, step):
        if self._run is not None:
            self._wandb.log({tag: value}, step=step)
