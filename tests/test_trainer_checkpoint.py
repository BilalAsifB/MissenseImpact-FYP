"""
Unit tests for epoch checkpointing and resume behavior in training/trainer.py.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from training.trainer import Trainer

sys.path.insert(0, str(Path(__file__).parent.parent))


class _TinyDataset(Dataset):
    def __init__(self):
        # Balanced labels to keep evaluation safe if triggered.
        self.x = torch.tensor(
            [
                [0.2, 0.1, -0.4, 1.1],
                [0.0, -1.0, 0.5, 0.2],
                [0.4, 0.3, 0.0, -0.2],
                [1.0, -0.1, 0.2, -0.7],
                [-0.3, 0.6, 0.1, 0.0],
                [0.7, 0.2, -0.8, 0.4],
            ],
            dtype=torch.float32,
        )
        self.y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "labels": self.y[idx]}


class _DummyESM(nn.Module):
    def __init__(self, hidden_layers=2):
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": hidden_layers})()
        self.encoder = type("Enc", (), {})()
        self.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(hidden_layers)])


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm = _DummyESM(hidden_layers=2)
        self.proj = nn.Linear(4, 4)

    def _freeze(self, freeze_layers):
        for i, layer in enumerate(self.esm.encoder.layer):
            req_grad = i >= freeze_layers
            for p in layer.parameters():
                p.requires_grad = req_grad

    def forward(self, x):
        return self.proj(x)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _DummyBackbone()
        self.fusion = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 1)

    def forward(self, batch):
        x = self.backbone(batch["x"])
        x = self.fusion(x)
        logit = self.classifier(x).squeeze(-1)
        return {"logit": logit}


def _loss_fn(logits, labels, weights=None):
    loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="none")
    if weights is not None:
        loss = loss * weights
    return loss


def _build_trainer(save_dir, model_id=0):
    model = _DummyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=_loss_fn,
        device="cpu",
        save_dir=str(save_dir),
        model_id=model_id,
        eval_every=10_000,
        scheduler=scheduler,
    )


def test_saves_checkpoint_every_epoch_and_resume_latest(tmp_path):
    train_loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False)
    val_loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False)

    trainer = _build_trainer(tmp_path, model_id=7)
    best = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=999,
        warmup_steps=0,
        start_epoch=1,
        max_epochs=2,
    )
    assert isinstance(best, float)

    ckpt1 = tmp_path / "model7_checkpoint_epoch_1.pt"
    ckpt2 = tmp_path / "model7_checkpoint_epoch_2.pt"
    assert ckpt1.exists()
    assert ckpt2.exists()

    trainer_resumed = _build_trainer(tmp_path, model_id=7)
    resumed_epoch, resumed_path = trainer_resumed.load_checkpoint("latest", map_location="cpu")
    assert resumed_epoch == 2
    assert resumed_path.name == "model7_checkpoint_epoch_2.pt"
    assert trainer_resumed.step == trainer.step
    assert trainer_resumed.optimizer.state_dict()["state"]
    assert trainer_resumed.scheduler.state_dict()["last_epoch"] == trainer.scheduler.state_dict()["last_epoch"]

    trainer_resumed.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=999,
        warmup_steps=0,
        start_epoch=resumed_epoch + 1,
        max_epochs=3,
    )
    ckpt3 = tmp_path / "model7_checkpoint_epoch_3.pt"
    assert ckpt3.exists()


def test_checkpoint_payload_contains_recovery_state(tmp_path):
    train_loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False)
    val_loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False)

    trainer = _build_trainer(tmp_path, model_id=3)
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=999,
        warmup_steps=0,
        start_epoch=1,
        max_epochs=1,
    )

    ckpt_path = tmp_path / "model3_checkpoint_epoch_1.pt"
    assert ckpt_path.exists()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    required_keys = {
        "model_state",
        "optimizer_state",
        "ema_state",
        "epoch",
        "step",
        "model_id",
        "best_auroc",
        "last_val_auroc",
        "train_loss",
        "metadata",
        "scheduler_state",
    }
    assert required_keys.issubset(set(ckpt.keys()))
    assert ckpt["epoch"] == 1
    assert ckpt["model_id"] == 3
    assert isinstance(ckpt["step"], int)
    assert "warmup_steps" in ckpt["metadata"]
    assert "max_steps" in ckpt["metadata"]
