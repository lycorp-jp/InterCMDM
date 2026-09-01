"""Load the plain-text configuration files written by the training scripts."""

from __future__ import annotations

import re
from argparse import Namespace
from pathlib import Path


_DECIMAL_PATTERN = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*)|(?:\d*\.\d+)|(?:\d+[eE][+-]?\d+)|"
    r"(?:\d+\.\d*[eE][+-]?\d+)|(?:\d*\.\d+[eE][+-]?\d+))$"
)
_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_OPTION_MARKERS = {
    "------------ Options -------------",
    "-------------- End ----------------",
}


def is_float(value) -> bool:
    """Return whether ``value`` is a decimal or scientific-notation scalar."""
    return _DECIMAL_PATTERN.fullmatch(str(value).strip()) is not None


def is_number(value) -> bool:
    """Return whether ``value`` is an integer scalar."""
    return _INTEGER_PATTERN.fullmatch(str(value).strip()) is not None


def _coerce_scalar(raw_value: str):
    if raw_value == "True":
        return True
    if raw_value == "False":
        return False
    if is_number(raw_value):
        return int(raw_value)
    if is_float(raw_value):
        return float(raw_value)
    return raw_value


def _read_options(opt_path) -> dict:
    parsed = {}
    path = Path(opt_path)
    print(f"Reading {path}")
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            text = line.strip()
            if not text or text in _OPTION_MARKERS:
                continue
            key, separator, value = text.partition(":")
            if not separator:
                raise ValueError(f"Malformed option at {path}:{line_number}: {text!r}")
            parsed[key.strip()] = _coerce_scalar(value.strip())
    return parsed


def _attach_experiment_paths(options) -> None:
    save_root = Path(options.checkpoints_dir) / options.dataset_name / options.name
    options.save_root = str(save_root)
    options.model_dir = str(save_root / "model")
    options.meta_dir = str(save_root / "meta")
    options.anim_dir = str(save_root / "animation")
    options.eval_dir = str(save_root / "eval")
    options.log_dir = str(save_root / "log")


def _attach_dataset_metadata(options) -> None:
    if options.dataset_name == "interhuman":
        options.data_root = "data/InterHuman"
        options.joints_num = 22
        return
    if options.dataset_name == "interx":
        options.data_root = "data/InterX"
        options.motion_dir = str(Path(options.data_root) / "motions")
        options.text_dir = str(Path(options.data_root) / "texts_processed")
        options.joints_num = 56
        options.max_motion_length = 150
        return
    raise KeyError(f"Dataset not recognized: {options.dataset_name}")


def get_opt(opt_path, device, complete=True, **kwargs):
    """Read an option file into a namespace and add runtime-only metadata."""
    values = _read_options(opt_path)
    options = Namespace(**values)
    options.device = device

    if complete:
        _attach_experiment_paths(options)
        _attach_dataset_metadata(options)
        options.is_train = False
        options.is_continue = False

    vars(options).update(kwargs)
    return options
