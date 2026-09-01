"""Small tensor utilities used by the conditional motion models.

The functions in this module intentionally stay stateless so they can be used
from both training and sampling code without constructing a helper object.
"""

from __future__ import annotations

import math
from functools import wraps
from typing import Callable, TypeVar

import torch
import torch.nn.functional as F


_T = TypeVar("_T")


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Return ``True`` for valid time steps in each sequence."""
    positions = torch.arange(max_len, device=lengths.device)
    return positions[None, :] < lengths[:, None]


def get_pad_mask_idx(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Build the key-padding mask shape expected by attention layers."""
    return seq.ne(pad_idx).unsqueeze(1)


def get_subsequent_mask(seq: torch.Tensor) -> torch.Tensor:
    """Build a lower-triangular causal mask for a batch of token sequences."""
    sequence_length = seq.shape[1]
    steps = torch.arange(sequence_length, device=seq.device)
    return steps[:, None].ge(steps[None, :]).unsqueeze(0)


def exists(value: object) -> bool:
    return value is not None


def default(value: _T | None, fallback: _T) -> _T:
    return value if value is not None else fallback


def eval_decorator(function: Callable[..., _T]) -> Callable[..., _T]:
    """Run a model method in eval mode and restore its original mode."""

    @wraps(function)
    def wrapped(model, *args, **kwargs):
        training_before_call = model.training
        model.eval()
        try:
            return function(model, *args, **kwargs)
        finally:
            model.train(training_before_call)

    return wrapped


def l2norm(tensor: torch.Tensor) -> torch.Tensor:
    return F.normalize(tensor, dim=-1)


def get_mask_subset_prob(mask: torch.Tensor, prob: float) -> torch.Tensor:
    """Randomly retain valid positions with probability ``prob``."""
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    draws = torch.rand(mask.shape, device=mask.device)
    return mask.bool() & draws.lt(prob)


def get_mask_special_tokens(
    ids: torch.Tensor, special_ids: list[int] | tuple[int, ...]
) -> torch.Tensor:
    """Mark entries in ``ids`` that match any special-token id."""
    if not special_ids:
        return torch.zeros_like(ids, dtype=torch.bool)
    special = torch.as_tensor(special_ids, device=ids.device, dtype=ids.dtype)
    return ids.unsqueeze(-1).eq(special).any(dim=-1)


def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    try:
        return {"relu": F.relu, "gelu": F.gelu}[activation]
    except KeyError as error:
        raise RuntimeError(
            f"activation must be 'relu' or 'gelu', got {activation!r}"
        ) from error


def uniform(shape, device=None, min=0, max=1) -> torch.Tensor:
    """Draw floating-point samples uniformly from ``[min, max)``."""
    return torch.empty(shape, device=device).uniform_(min, max)


def prob_mask_like(shape, prob: float, device=None) -> torch.Tensor:
    """Create an independent Bernoulli mask without materializing probabilities."""
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"prob must be in [0, 1], got {prob}")
    if prob == 0.0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    if prob == 1.0:
        return torch.ones(shape, device=device, dtype=torch.bool)
    return torch.rand(shape, device=device).lt(prob)


def log(tensor: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return tensor.clamp_min(eps).log()


def gumbel_noise(tensor: torch.Tensor) -> torch.Tensor:
    """Draw standard Gumbel noise matching ``tensor``'s shape and device."""
    exponential = torch.empty_like(tensor).exponential_()
    return -exponential.log()


def gumbel_sample(
    tensor: torch.Tensor, temperature: float = 1.0, dim: int = 1
) -> torch.Tensor:
    safe_temperature = max(temperature, 1e-10)
    return (tensor / safe_temperature + gumbel_noise(tensor)).argmax(dim=dim)


def top_k(logits: torch.Tensor, thres: float = 0.9, dim: int = 1) -> torch.Tensor:
    """Keep the largest fraction of logits and set all others to ``-inf``."""
    keep_count = math.ceil((1.0 - thres) * logits.shape[dim])
    values, indices = logits.topk(keep_count, dim=dim)
    filtered = logits.new_full(logits.shape, float("-inf"))
    return filtered.scatter(dim, indices, values)


def cosine_schedule(tensor: torch.Tensor) -> torch.Tensor:
    return torch.cos(tensor * (math.pi / 2.0))


def cosine_schedule_backward(tensor: torch.Tensor) -> torch.Tensor:
    return torch.acos(tensor) / (math.pi / 2.0)


def scale_cosine_schedule(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    scheduled = 1.0 + scale * (cosine_schedule(tensor) - 1.0)
    return scheduled.clamp(0.0, 1.0)


def q_schedule(bs: int, low: int, high: int, device) -> torch.Tensor:
    """Sample integer corruption counts according to a cosine schedule."""
    progress = 1.0 - cosine_schedule(torch.rand(bs, device=device))
    return (progress * (high - low - 1)).round().long().add(low)


def cal_performance(
    pred: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int | None = None,
    smoothing: float = 0.0,
    tk: int = 1,
):
    """Return cross-entropy loss, top-1 predictions, and top-k accuracy."""
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    top_predictions = pred.topk(k=tk, dim=1).indices
    valid = (
        torch.ones_like(labels, dtype=torch.bool)
        if ignore_index is None
        else labels.ne(ignore_index)
    )
    correct = top_predictions.eq(labels.unsqueeze(1)).any(dim=1)
    valid_correct = correct.masked_select(valid)
    accuracy = valid_correct.float().mean().item() if valid_correct.numel() else 0.0
    return loss, top_predictions[:, 0], accuracy


def cal_loss(
    pred: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int | None = None,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute class loss with optional explicit label smoothing."""
    if not smoothing:
        if ignore_index is None:
            return F.cross_entropy(pred, labels)
        return F.cross_entropy(pred, labels, ignore_index=ignore_index)

    class_count = pred.size(1)
    valid = (
        torch.ones_like(labels, dtype=torch.bool)
        if ignore_index is None
        else labels.ne(ignore_index)
    )
    safe_labels = labels.masked_fill(~valid, 0)
    targets = F.one_hot(safe_labels, num_classes=class_count).movedim(-1, 1)
    targets = targets.to(dtype=pred.dtype)
    targets = targets * (1.0 - smoothing) + (1.0 - targets) * smoothing / (
        class_count - 1
    )
    per_item = (targets * -F.log_softmax(pred, dim=1)).sum(dim=1)
    selected = per_item.masked_select(valid)
    return selected.mean() if selected.numel() else pred.sum() * 0.0
