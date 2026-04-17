"""
EMA checkpointing + training loop.
"""

from __future__ import annotations
import logging
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


class EMAModel:
    """Exponential moving average of model weights (AM paper: decay=0.999)."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            n: p.data.clone().float()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self._backup = {}

    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n] = (
                    self.decay * self.shadow[n]
                    + (1 - self.decay) * p.data.float()
                )

    def apply_to(self, model: nn.Module):
        self._backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self._backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n].to(p.dtype))

    def restore(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup = {}

    def state_dict(self):
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, d):
        self.shadow = d["shadow"]
        self.decay = d.get("decay", self.decay)


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = 0.0
        self.wait = 0
        self.best_step = 0

    def step(self, metric: float, step: int) -> bool:
        if metric > self.best + self.min_delta:
            self.best = metric
            self.wait = 0
            self.best_step = step
        else:
            self.wait += 1
        if self.wait >= self.patience:
            log.info(
                "Early stop at step %d (best=%.4f at step %d)",
                step, self.best, self.best_step,
            )
            return True
        return False


class Trainer:
    """Wraps training loop with EMA, early stopping, and checkpointing."""

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device: str,
        save_dir: str,
        model_id: int = 0,
        ema_decay: float = 0.999,
        patience: int = 15,
        log_every: int = 100,
        eval_every: int = 500,
        scheduler=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = model_id
        self.ema = EMAModel(model, ema_decay)
        self.stopper = EarlyStopping(patience)
        self.log_every = log_every
        self.eval_every = eval_every
        self.scheduler = scheduler
        self.step = 0
        self.best_auroc = 0.0
        self.last_val_auroc = None
        # AMP Scaler
        self.scaler = GradScaler(self.device, enabled=True)
        # TF32 boost
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Store the configured freeze depth so we can re-apply it after warmup
        self._freeze_layers = model.backbone.esm.config.num_hidden_layers
        for i, layer in enumerate(model.backbone.esm.encoder.layer):
            if not any(p.requires_grad for p in layer.parameters()):
                self._freeze_layers = i + 1  # first frozen layer index + 1
                break

    def _to(self, batch):
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def train_step(self, batch: dict) -> float:
        self.model.train()
        batch = self._to(batch)
        
        with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=True):
            out = self.model(batch)
            loss = self.loss_fn(
                out["logit"], batch["labels"],
                weights=batch.get("weights"),
            ).mean()
        
        self.optimizer.zero_grad()
        # Scaled backward
        self.scaler.scale(loss).backward()
        # Unscale before clipping to avoid affecting the clipping threshold
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        # Step with scaled gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.scheduler:
            self.scheduler.step()
        
        self.ema.update(self.model)
        self.step += 1
        
        return loss.item()

    @torch.no_grad()
    def evaluate(self, val_loader) -> float:
        self.model.eval()
        logits, labels = [], []
        self.ema.apply_to(self.model)
        for batch in val_loader:
            batch = self._to(batch)
            logits.append(self.model(batch)["logit"].cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
        self.ema.restore(self.model)
        lg = np.concatenate(logits)
        lb = np.concatenate(labels)
        return float(roc_auc_score(lb, lg)) if len(np.unique(lb)) > 1 else 0.5

    def _checkpoint_name(self, epoch: int) -> str:
        return f"model{self.model_id}_checkpoint_epoch_{epoch}.pt"

    def latest_checkpoint_path(self):
        pattern = f"model{self.model_id}_checkpoint_epoch_*.pt"
        candidates = sorted(
            self.save_dir.glob(pattern),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        return candidates[-1] if candidates else None

    def save_checkpoint(
        self,
        epoch: int,
        train_loss: float | None = None,
        val_auroc: float | None = None,
        metadata: dict | None = None,
    ):
        path = self.save_dir / self._checkpoint_name(epoch)
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "ema_state": self.ema.state_dict(),
            "epoch": epoch,
            "step": self.step,
            "model_id": self.model_id,
            "best_auroc": self.best_auroc,
            "last_val_auroc": val_auroc,
            "train_loss": train_loss,
            "metadata": metadata or {},
        }
        if self.scheduler is not None:
            payload["scheduler_state"] = self.scheduler.state_dict()

        # Write atomically to avoid leaving partial checkpoint files.
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        log.info(
            "Checkpoint saved: %s (epoch=%d, step=%d, val_auroc=%s)",
            path.name,
            epoch,
            self.step,
            "n/a" if val_auroc is None else f"{val_auroc:.4f}",
        )
        return path

    def load_checkpoint(
        self,
        checkpoint_path: str | Path | None = None,
        map_location: str | None = None,
        strict: bool = True,
    ):
        if checkpoint_path is None or str(checkpoint_path).lower() == "latest":
            resolved = self.latest_checkpoint_path()
            if resolved is None:
                raise FileNotFoundError(
                    f"No checkpoint found in {self.save_dir} for model_id={self.model_id}"
                )
            checkpoint_path = resolved

        ckpt_path = Path(checkpoint_path)
        ckpt = torch.load(ckpt_path, map_location=map_location or self.device)

        model_state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(model_state, strict=strict)

        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_state" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state"])
        if "scheduler_state" in ckpt and self.scheduler is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        if "ema_state" in ckpt:
            self.ema.load_state_dict(ckpt["ema_state"])

        self.step = int(ckpt.get("step", 0))
        self.best_auroc = float(ckpt.get("best_auroc", 0.0))
        self.last_val_auroc = ckpt.get("last_val_auroc", None)
        epoch = int(ckpt.get("epoch", 0))

        log.info(
            "Resumed from %s (epoch=%d, step=%d, best_auroc=%.4f)",
            ckpt_path.name,
            epoch,
            self.step,
            self.best_auroc,
        )
        return epoch, ckpt_path

    def fit(
        self,
        train_loader,
        val_loader,
        max_steps: int = 350_000,
        warmup_steps: int = 1000,
        start_epoch: int = 1,
        max_epochs: int | None = None,
    ) -> float:
        """
        Head warmup then full training, mirroring AM paper protocol.

        Phase 1 (warmup_steps): backbone fully frozen, only head trains.
        Phase 2 (remainder):    re-freeze bottom N layers per the original
                                freeze_layers config, unfreeze top layers.
        """
        steps_per_epoch = len(train_loader)
        if max_epochs is None:
            max_epochs = max(1, int(math.ceil(max_steps / steps_per_epoch)))

        in_warmup = self.step < warmup_steps
        if in_warmup:
            for p in self.model.backbone.parameters():
                p.requires_grad = False
            log.info("Head warmup active (until step %d)", warmup_steps)
        else:
            for p in self.model.backbone.parameters():
                p.requires_grad = True
            self.model.backbone._freeze(self._freeze_layers)
            log.info(
                "Full training resumed (bottom %d ESM layers frozen)",
                self._freeze_layers,
            )

        should_stop = False
        for epoch in range(start_epoch, max_epochs + 1):
            if should_stop:
                break

            epoch_losses = []
            for batch in train_loader:
                if in_warmup and self.step >= warmup_steps:
                    # Warmup boundary reached: switch to main training setup.
                    for p in self.model.backbone.parameters():
                        p.requires_grad = True
                    self.model.backbone._freeze(self._freeze_layers)
                    in_warmup = False
                    log.info(
                        "Switched to full training at step %d (bottom %d ESM layers frozen)",
                        self.step,
                        self._freeze_layers,
                    )

                loss = self.train_step(batch)
                epoch_losses.append(loss)

                if self.step % self.log_every == 0:
                    log.info("step %d  loss=%.4f", self.step, loss)

                if self.step % self.eval_every == 0:
                    auroc = self.evaluate(val_loader)
                    self.last_val_auroc = auroc
                    log.info("step %d  val_auroc=%.4f", self.step, auroc)
                    if auroc > self.best_auroc:
                        self.best_auroc = auroc
                    if self.stopper.step(auroc, self.step):
                        should_stop = True
                        break

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            self.save_checkpoint(
                epoch=epoch,
                train_loss=epoch_loss,
                val_auroc=self.last_val_auroc,
                metadata={
                    "warmup_steps": warmup_steps,
                    "max_steps": max_steps,
                },
            )

            if should_stop:
                break

        return self.best_auroc


def _inf(loader):
    while True:
        yield from loader
