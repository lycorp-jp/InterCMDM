"""Command-line configuration for training the temporal autoencoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


_OPTIONS_HEADER = "------------ Options -------------"
_OPTIONS_FOOTER = "-------------- End ----------------"


def str2bool(value):
    """Parse common textual boolean spellings for ``argparse``."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"yes", "true", "t", "y", "1"}:
        return True
    if normalized in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, received {value!r}.")


def _add_data_options(parser):
    group = parser.add_argument_group("data")
    group.add_argument("--dataset_name", default="interhuman", help="dataset directory")
    group.add_argument(
        "--motion_rep", default="smpl", help="how the motion is represented"
    )
    group.add_argument("--batch_size", type=int, default=256, help="batch size")
    group.add_argument(
        "--window_size", type=int, default=64, help="training motion length"
    )
    group.add_argument(
        "--window_stride",
        type=int,
        default=10,
        help="stride used to sample motion windows",
    )
    group.add_argument("--gpu_id", type=int, default=0, help="GPU id")
    group.add_argument("--cache", type=str2bool, default=True, help="cache the dataset")
    group.add_argument("--feature_dim", type=int, default=262, help="feature dimension")


def _add_optimization_options(parser):
    group = parser.add_argument_group("optimization")
    group.add_argument(
        "--max_epoch",
        type=int,
        default=50,
        help="number of total epochs to run",
    )
    group.add_argument("--lr", type=float, default=2e-4, help="max learning rate")
    group.add_argument("--gamma", type=float, default=0.1, help="learning rate decay")
    group.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    group.add_argument(
        "--commit",
        type=float,
        default=0.02,
        help="weight of the commitment loss",
    )
    group.add_argument(
        "--recons_loss",
        default="l1_smooth",
        help="reconstruction loss implementation",
    )
    group.add_argument(
        "--loss_explicit",
        type=float,
        default=1,
        help="weight of the explicit loss",
    )
    group.add_argument(
        "--loss_vel", type=float, default=100, help="weight of the velocity loss"
    )
    group.add_argument(
        "--loss_bn", type=float, default=5, help="weight of the bone-length loss"
    )
    group.add_argument(
        "--loss_geo",
        type=float,
        default=0.01,
        help="weight of the geodesic loss",
    )
    group.add_argument(
        "--loss_fc",
        type=float,
        default=500,
        help="weight of the foot-contact loss",
    )


def _add_model_options(parser):
    group = parser.add_argument_group("autoencoder")
    group.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    group.add_argument("--nb_code", type=int, default=1024, help="number of embeddings")
    group.add_argument(
        "--mu",
        type=float,
        default=0.99,
        help="EMA coefficient for codebook updates",
    )
    group.add_argument(
        "--down_t", type=int, default=2, help="number of downsampling stages"
    )
    group.add_argument("--stride_t", type=int, default=2, help="temporal stride size")
    group.add_argument("--width", type=int, default=512, help="network width")
    group.add_argument("--depth", type=int, default=2, help="residual blocks per stage")
    group.add_argument(
        "--dilation_growth_rate",
        type=int,
        default=3,
        help="dilation growth rate",
    )
    group.add_argument(
        "--output_emb_width",
        type=int,
        default=64,
        help="output embedding width",
    )
    group.add_argument(
        "--vq_act",
        default="relu",
        choices=("relu", "silu", "gelu"),
        help="activation function",
    )
    group.add_argument("--vq_norm", default=None, help="normalization layer")
    group.add_argument(
        "--num_quantizers", type=int, default=1, help="number of quantizers"
    )
    group.add_argument("--shared_codebook", action="store_true")
    group.add_argument(
        "--quantize_dropout_prob",
        type=float,
        default=0.2,
        help="quantizer dropout probability",
    )


def _add_runtime_options(parser):
    group = parser.add_argument_group("runtime")
    group.add_argument("--name", default="vq_default", help="trial name")
    group.add_argument("--is_continue", action="store_true", help="resume the trial")
    group.add_argument(
        "--checkpoints_dir",
        default="./checkpoints",
        help="directory used for model artifacts",
    )
    group.add_argument(
        "--save_every_e",
        type=int,
        default=1,
        help="checkpoint interval in epochs",
    )
    group.add_argument(
        "--eval_every_e",
        type=int,
        default=1,
        help="evaluation interval in epochs",
    )
    group.add_argument(
        "--feat_bias", type=float, default=5, help="feature scaling bias"
    )
    group.add_argument(
        "--do_eval", action="store_true", help="evaluate during training"
    )
    group.add_argument(
        "--test_batch_size",
        type=int,
        default=96,
        help="evaluation batch size",
    )
    group.add_argument(
        "--project_name", default="InterCMDM", help="Weights & Biases project"
    )
    group.add_argument("--entity", default="user_name", help="Weights & Biases entity")


def _build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_data_options(parser)
    _add_optimization_options(parser)
    _add_model_options(parser)
    _add_runtime_options(parser)
    return parser


def _option_lines(values):
    yield _OPTIONS_HEADER
    for name, value in sorted(values.items()):
        yield f"{name}: {value}"
    yield _OPTIONS_FOOTER


def _write_training_options(values, destination):
    destination.mkdir(parents=True, exist_ok=True)
    content = "\n".join(_option_lines(values)) + "\n"
    (destination / "opt.txt").write_text(content, encoding="utf-8")


def arg_parse(is_train=False):
    """Parse VQ/TAE arguments and optionally persist the training config."""
    options = _build_parser().parse_args()
    if options.gpu_id != -1:
        torch.cuda.set_device(options.gpu_id)

    values = vars(options).copy()
    print("\n".join(_option_lines(values)))
    options.is_train = is_train

    if is_train:
        experiment_dir = (
            Path(options.checkpoints_dir) / options.dataset_name / options.name
        )
        _write_training_options(values, experiment_dir)
    return options
