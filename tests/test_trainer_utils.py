"""
Additional unit tests for training/trainer.py utilities and edge behavior.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from training.trainer import EMAModel, EarlyStopping, Trainer


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=False)

    def forward(self, batch):
        x = batch["x"].float().view(-1)
        return {"logit": x * self.w}


class _DummyESM(nn.Module):
    def __init__(self, hidden_layers=2):
        super().__init__()
        self.config = type("Cfg", (), {"num_hidden_layers": hidden_layers})()
        self.encoder = type("Enc", (), {})()
        self.encoder.layer = nn.ModuleList([nn.Linear(1, 1) for _ in range(hidden_layers)])


class _DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.esm = _DummyESM(hidden_layers=2)
        self.proj = nn.Linear(1, 1)

    def _freeze(self, freeze_layers):
        for i, layer in enumerate(self.esm.encoder.layer):
            req_grad = i >= freeze_layers
            for p in layer.parameters():
                p.requires_grad = req_grad

    def forward(self, x):
        return self.proj(x)


class _TrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _DummyBackbone()
        self.head = nn.Linear(1, 1)

    def forward(self, batch):
        x = batch["x"].float().view(-1, 1)
        z = self.backbone(x)
        return {"logit": self.head(z).squeeze(-1)}


def _loss(logits, labels, weights=None):
    loss = (logits - labels.float()) ** 2
    if weights is not None:
        loss = loss * weights
    return loss


def _make_trainer(tmp_path):
    model = _TrainModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    return Trainer(
        model=model,
        optimizer=opt,
        loss_fn=_loss,
        device="cpu",
        save_dir=str(tmp_path),
        eval_every=10_000,
    )


def test_ema_tracks_only_trainable_parameters():
    model = _TinyModel()
    ema = EMAModel(model, decay=0.5)
    assert "w" in ema.shadow
    assert "b" not in ema.shadow


def test_ema_update_apply_restore_roundtrip():
    model = _TinyModel()
    ema = EMAModel(model, decay=0.5)
    model.w.data.fill_(3.0)
    ema.update(model)
    # previous shadow was 1.0 => new shadow is 0.5*1 + 0.5*3 = 2.0
    assert torch.allclose(ema.shadow["w"], torch.tensor([2.0]))

    model.w.data.fill_(10.0)
    ema.apply_to(model)
    assert torch.allclose(model.w.data, torch.tensor([2.0]))
    ema.restore(model)
    assert torch.allclose(model.w.data, torch.tensor([10.0]))


def test_early_stopping_patience_logic():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert not stopper.step(0.5, 1)   # improvement
    assert stopper.best == 0.5
    assert not stopper.step(0.505, 2)  # below min_delta -> wait 1
    assert stopper.wait == 1
    assert stopper.step(0.506, 3)      # wait 2 -> stop


def test_latest_checkpoint_path_returns_highest_epoch(tmp_path):
    trainer = _make_trainer(tmp_path)
    (tmp_path / "model0_checkpoint_epoch_1.pt").write_text("x")
    (tmp_path / "model0_checkpoint_epoch_3.pt").write_text("x")
    (tmp_path / "model0_checkpoint_epoch_2.pt").write_text("x")
    latest = trainer.latest_checkpoint_path()
    assert latest.name == "model0_checkpoint_epoch_3.pt"


def test_to_moves_tensor_fields_to_target_device(tmp_path):
    trainer = _make_trainer(tmp_path)
    batch = {"x": torch.tensor([1.0]), "labels": torch.tensor([0.0]), "meta": "keep"}
    out = trainer._to(batch)
    assert out["x"].device.type == "cpu"
    assert out["labels"].device.type == "cpu"
    assert out["meta"] == "keep"


def test_evaluate_returns_half_when_labels_single_class(monkeypatch, tmp_path):
    trainer = _make_trainer(tmp_path)

    def _loader():
        for _ in range(2):
            yield {"x": torch.tensor([1.0]), "labels": torch.tensor([1.0])}

    monkeypatch.setattr(trainer.model, "forward", lambda batch: {"logit": torch.tensor([0.1])})
    auroc = trainer.evaluate(_loader())
    assert auroc == 0.5


def test_load_checkpoint_raises_when_latest_missing(tmp_path):
    trainer = _make_trainer(tmp_path)
    try:
        trainer.load_checkpoint("latest")
        raise AssertionError("Expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_fit_crosses_warmup_boundary_and_calls_freeze(monkeypatch, tmp_path):
    trainer = _make_trainer(tmp_path)
    freeze_called = {"value": None}
    original_freeze = trainer.model.backbone._freeze

    def _wrapped_freeze(n):
        freeze_called["value"] = n
        return original_freeze(n)

    monkeypatch.setattr(trainer.model.backbone, "_freeze", _wrapped_freeze)

    batch = {
        "x": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "labels": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
    }
    train_loader = [batch, batch]
    val_loader = [batch]

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=20,
        warmup_steps=1,
        start_epoch=1,
        max_epochs=1,
    )

    assert freeze_called["value"] == trainer._freeze_layers
    assert (Path(tmp_path) / "model0_checkpoint_epoch_1.pt").exists()
