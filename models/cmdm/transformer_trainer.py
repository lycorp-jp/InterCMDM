"""Training orchestration for the interaction-conditioned DiT model."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch import optim

from eval_inter_cmdm import evaluation_during_training
from utils.utils import WandbLogger, print_current_loss


def def_value() -> float:
    """Factory used by the rolling metric accumulator."""
    return 0.0


@torch.no_grad()
def update_ema(model, ema_model, ema_decay: float) -> None:
    """Move ``ema_model`` parameters toward the current model parameters."""
    model_parameters = model.parameters()
    for averaged, current in zip(ema_model.parameters(), model_parameters):
        averaged.mul_(ema_decay).add_(current, alpha=1.0 - ema_decay)


class DiTTransformerTrainer:
    """Own optimization, checkpointing, validation, and evaluation for DiT."""

    _MILESTONE_FRACTIONS = (0.50, 0.70, 0.85)

    def __init__(self, args, dit_model, tae_model):
        self.opt = args
        self.device = args.device
        self.dit_model = dit_model
        self.tae_model = tae_model
        self.tae_model.eval()

        self.ema_decay = getattr(args, "ema_decay", 0.9999)
        self.ema_dit_model = self._build_ema_model(dit_model)
        if args.is_train:
            self.logger = WandbLogger(args)

    @staticmethod
    def _build_ema_model(model):
        averaged_model = copy.deepcopy(model)
        averaged_model.requires_grad_(False)
        return averaged_model

    @staticmethod
    def _strip_clip_weights(state_dict):
        return {
            name: value
            for name, value in state_dict.items()
            if not name.startswith("clip_")
        }

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        """Linearly ramp the optimizer learning rate during warm-up."""
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for group in self.opt_dit_model.param_groups:
            group["lr"] = current_lr
        return current_lr

    def _unpack_batch(self, batch_data):
        dataset_name = self.opt.dataset_name
        if dataset_name == "interhuman":
            _, conditions, first_motion, second_motion, lengths = batch_data
        elif dataset_name == "interx":
            _, _, conditions, _, paired_motion, lengths, _ = batch_data
            first_motion, second_motion = paired_motion.split(6, dim=-1)
            first_motion = first_motion.flatten(start_dim=-2)
            second_motion = second_motion.flatten(start_dim=-2)
        else:
            raise KeyError(f"Unsupported dataset: {dataset_name}")
        return conditions, first_motion, second_motion, lengths

    def _encode_motion(self, motion):
        motion = rearrange(motion, "b l d -> b l 1 d")
        encoded = self.tae_model.encode(motion)
        return rearrange(
            encoded,
            f"{self.tae_model.encode_dim} -> b l 1 d",
            d=self.tae_model.output_emb_width,
        )

    def forward(self, batch_data):
        conditions, first_motion, second_motion, lengths = self._unpack_batch(
            batch_data
        )
        first_motion = first_motion.detach().to(self.device, dtype=torch.float32)
        second_motion = second_motion.detach().to(self.device, dtype=torch.float32)
        lengths = lengths.detach().to(self.device, dtype=torch.long)

        with torch.no_grad():
            first_latent = self._encode_motion(first_motion)
            second_latent = self._encode_motion(second_motion)
        paired_latent = torch.cat((first_latent, second_latent), dim=1)

        if torch.is_tensor(conditions):
            conditions = conditions.to(self.device, dtype=torch.float32)
        latent_lengths = torch.div(lengths, 4, rounding_mode="floor")
        return self.dit_model.forward_loss(paired_latent, conditions, latent_lengths)

    def update(self, batch_data):
        """Run one optimizer and EMA update, returning a scalar loss."""
        loss = self.forward(batch_data)
        self.opt_dit_model.zero_grad()
        loss.backward()
        self.opt_dit_model.step()
        self.scheduler.step()
        update_ema(self.dit_model, self.ema_dit_model, self.ema_decay)
        return loss.item()

    def save(self, file_name, ep, total_it):
        checkpoint = {
            "dit_model": self._strip_clip_weights(self.dit_model.state_dict()),
            "ema_dit_model": self._strip_clip_weights(self.ema_dit_model.state_dict()),
            "opt_dit_model": self.opt_dit_model.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "ep": ep,
            "total_it": total_it,
        }
        torch.save(checkpoint, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing, unexpected = self.dit_model.load_state_dict(
            checkpoint["dit_model"], strict=False
        )
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
        invalid_missing = [name for name in missing if not name.startswith("clip_")]
        if invalid_missing:
            raise RuntimeError(f"Missing checkpoint keys: {invalid_missing}")

        ema_state = checkpoint.get("ema_dit_model")
        if ema_state is None:
            self.ema_dit_model = self._build_ema_model(self.dit_model)
        else:
            self.ema_dit_model.load_state_dict(ema_state, strict=False)

        self.opt_dit_model.load_state_dict(checkpoint["opt_dit_model"])
        scheduler_state = checkpoint.get("scheduler")
        if scheduler_state is not None:
            progress = {
                key: scheduler_state[key]
                for key in ("last_epoch", "_step_count")
                if key in scheduler_state
            }
            if progress:
                self.scheduler.load_state_dict(progress)
        else:
            print("Resume without scheduler state")
        return checkpoint["ep"], checkpoint["total_it"]

    def _configure_run(self, train_batches: int) -> int:
        if train_batches <= 0:
            raise ValueError("train_loader must contain at least one batch")

        total_iterations = self.opt.max_epoch * train_batches
        self.opt.milestones = [
            int(total_iterations * fraction) for fraction in self._MILESTONE_FRACTIONS
        ]
        self.opt.warm_up_iter = max(1, train_batches // 4)
        self.opt.log_every = max(1, train_batches // 10)
        self.opt.save_latest = max(1, train_batches // 2)
        return total_iterations

    def _create_optimizer(self):
        self.opt_dit_model = optim.AdamW(
            self.dit_model.parameters(),
            betas=(0.9, 0.99),
            lr=self.opt.lr,
            weight_decay=1e-5,
        )
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.opt_dit_model,
            milestones=self.opt.milestones,
            gamma=self.opt.gamma,
        )

    def _print_run_summary(self, total_iterations, train_batches, val_batches):
        print(f"Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iterations}")
        print(
            f"Iters Per Epoch, Training: {train_batches:04d}, "
            f"Validation: {val_batches:03d}"
        )
        print(f"Milestones: {self.opt.milestones}")
        print(
            f"Warm Up Iterations: {self.opt.warm_up_iter:04d}, "
            f"Log Every: {self.opt.log_every:04d}, "
            f"Save Latest: {self.opt.save_latest:04d}"
        )

    def _flush_training_metrics(
        self, metrics, start_time, iteration, total_iterations, epoch, inner_iter
    ):
        means = OrderedDict(
            (name, value / self.opt.log_every) for name, value in metrics.items()
        )
        for name, value in means.items():
            self.logger.add_scalar(f"Train/{name}", value, iteration)
        print_current_loss(
            start_time,
            iteration,
            total_iterations,
            means,
            epoch=epoch,
            inner_iter=inner_iter,
        )

    def _validation_loss(self, val_loader) -> float:
        self.tae_model.eval()
        self.dit_model.eval()
        losses = []
        with torch.no_grad():
            for batch_data in val_loader:
                losses.append(self.forward(batch_data).item())
        if not losses:
            raise ValueError("val_loader must contain at least one batch")
        return float(np.mean(losses))

    def _run_evaluation(
        self, test_loader, eval_wrapper, epoch, iteration, eval_file, best
    ):
        self.tae_model.eval()
        self.dit_model.eval()
        fid, matching, top1 = evaluation_during_training(
            self.opt,
            self.tae_model,
            test_loader,
            eval_wrapper,
            epoch,
            eval_file,
            trans=self.dit_model,
        )
        self.logger.add_scalar("Test/FID", fid, epoch)
        self.logger.add_scalar("Test/Matching", matching, epoch)
        self.logger.add_scalar("Test/Top1", top1, epoch)

        if fid < best["fid"]:
            best["fid"] = fid
            self.save(Path(self.opt.model_dir) / "best_fid.tar", epoch, iteration)
            print("Best FID Model So Far!~")
        if top1 > best["top1"]:
            best["top1"] = top1
            self.save(Path(self.opt.model_dir) / "best_top1.tar", epoch, iteration)
            print("Best Top1 Model So Far!~")

    def train(self, train_loader, val_loader, test_loader, eval_wrapper):
        self.dit_model.to(self.device)
        self.tae_model.to(self.device)
        self.ema_dit_model.to(self.device)

        train_batches = len(train_loader)
        total_iterations = self._configure_run(train_batches)
        self._print_run_summary(total_iterations, train_batches, len(val_loader))
        self._create_optimizer()

        epoch = 0
        iteration = 0
        if self.opt.is_continue:
            epoch, iteration = self.resume(Path(self.opt.model_dir) / "latest.tar")
            iteration -= iteration % self.opt.log_every
            print(f"Load model epoch:{epoch:d} iterations:{iteration:d}")

        started_at = time.time()
        metrics = defaultdict(def_value, OrderedDict())
        best = {"loss": np.inf, "fid": np.inf, "top1": -np.inf}
        eval_file = (
            Path(self.opt.eval_dir) / "evaluation_training.log"
            if self.opt.do_eval
            else None
        )

        while epoch < self.opt.max_epoch:
            epoch += 1
            self.dit_model.train()
            self.tae_model.eval()
            if epoch > 200:
                self.opt.eval_every_e = 4

            for inner_iter, batch in enumerate(train_loader):
                iteration += 1
                if iteration < self.opt.warm_up_iter:
                    self.update_lr_warm_up(
                        iteration, self.opt.warm_up_iter, self.opt.lr
                    )

                metrics["loss"] += self.update(batch)
                metrics["lr"] += self.opt_dit_model.param_groups[0]["lr"]

                if iteration % self.opt.log_every == 0:
                    self._flush_training_metrics(
                        metrics,
                        started_at,
                        iteration,
                        total_iterations,
                        epoch,
                        inner_iter,
                    )
                    metrics = defaultdict(def_value, OrderedDict())

                if iteration % self.opt.save_latest == 0:
                    self.save(
                        Path(self.opt.model_dir) / "latest.tar",
                        epoch,
                        iteration,
                    )

            self.save(Path(self.opt.model_dir) / "latest.tar", epoch, iteration)
            print("Validation time:")
            validation_loss = self._validation_loss(val_loader)
            print(f"Validation loss:{validation_loss:.3f}")
            self.logger.add_scalar("Val/loss", validation_loss, iteration)

            if validation_loss < best["loss"]:
                print(f"Improved Loss from {best['loss']:.02f} to {validation_loss}!!!")
                best["loss"] = validation_loss
                self.save(
                    Path(self.opt.model_dir) / "finest.tar",
                    epoch,
                    iteration,
                )

            should_evaluate = self.opt.do_eval and epoch % self.opt.eval_every_e == 0
            if should_evaluate:
                self._run_evaluation(
                    test_loader,
                    eval_wrapper,
                    epoch,
                    iteration,
                    eval_file,
                    best,
                )
            print()
