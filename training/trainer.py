"""
EMA checkpointing + training loop.
"""

from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
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
        self.model = model
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
        out = self.model(batch)
        loss = self.loss_fn(
            out["logit"], batch["labels"],
            weights=batch.get("weights"),
        ).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()
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

    def save(self, auroc: float):
        path = self.save_dir / f"model{self.model_id}_step{self.step}.pt"
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "ema_state": self.ema.state_dict(),
                "step": self.step,
                "val_auroc": auroc,
                "model_id": self.model_id,
            },
            path,
        )
        log.info("Checkpoint: %s  auROC=%.4f", path.name, auroc)
        return path

    def fit(
        self,
        train_loader,
        val_loader,
        max_steps: int = 350_000,
        warmup_steps: int = 1000,
    ) -> float:
        """
        Head warmup then full training, mirroring AM paper protocol.

        Phase 1 (warmup_steps): backbone fully frozen, only head trains.
        Phase 2 (remainder):    re-freeze bottom N layers per the original
                                freeze_layers config, unfreeze top layers.
        """
        # Phase 1: freeze entire backbone
        for p in self.model.backbone.parameters():
            p.requires_grad = False
        log.info("Head warmup: %d steps", warmup_steps)
        for batch in _inf(train_loader):
            if self.step >= warmup_steps:
                break
            self.train_step(batch)

        # Phase 2: re-apply the original freeze depth.
        # Re-enable all backbone params first, then freeze from the bottom.
        for p in self.model.backbone.parameters():
            p.requires_grad = True
        self.model.backbone._freeze(self._freeze_layers)
        log.info(
            "Full training: up to %d steps (bottom %d ESM layers frozen)",
            max_steps, self._freeze_layers,
        )

        best_auroc = 0.0
        for batch in _inf(train_loader):
            if self.step >= max_steps:
                break
            loss = self.train_step(batch)
            if self.step % self.log_every == 0:
                log.info("step %d  loss=%.4f", self.step, loss)
            if self.step % self.eval_every == 0:
                auroc = self.evaluate(val_loader)
                log.info("step %d  val_auroc=%.4f", self.step, auroc)
                if auroc > best_auroc:
                    best_auroc = auroc
                    self.save(auroc)
                if self.stopper.step(auroc, self.step):
                    break
        return best_auroc


def _inf(loader):
    while True:
        yield from loader
