#!/usr/bin/env python3
"""
Train 3 independent models (AM protocol).
"""

from training.trainer import Trainer
from training.loss import clipped_sigmoid_xent
from model.esm_missense import ESMMissense
from data.dataset import SASVariantDataset, collate_variants
import sys
import json
import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")


def train_one(model_id, cfg, train_csv, val_csv, save_dir, device):
    torch.manual_seed(cfg.get("seed", 42) + model_id)

    train_dl = DataLoader(
        SASVariantDataset(train_csv), batch_size=cfg.get("batch_size", 16),
        shuffle=True, collate_fn=collate_variants,
        num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(
        SASVariantDataset(val_csv), batch_size=32,
        shuffle=False, collate_fn=collate_variants, num_workers=2)

    model = ESMMissense(
        freeze_esm_layers=cfg.get("freeze_esm_layers", 30),
        proj_dim=cfg.get("proj_dim", 512),
        hidden_dim=cfg.get("hidden_dim", 256),
        dropout=cfg.get("dropout", 0.1),
    ).to(device)

    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    hd = list(model.fusion.parameters()) + list(model.classifier.parameters())
    opt = torch.optim.AdamW(
        [{"params": bb, "lr": cfg.get("esm_lr", 1e-5)},
         {"params": hd, "lr": cfg.get("head_lr", 1e-4)}],
        weight_decay=cfg.get("weight_decay", 0.01), eps=1e-5)

    cn, cp = cfg.get("clip_neg", 0.0), cfg.get("clip_pos", -1.0)

    def loss_fn(logits, labels, weights=None):
        return clipped_sigmoid_xent(logits, labels, cn, cp, weights)

    trainer = Trainer(model=model, optimizer=opt, loss_fn=loss_fn,
                      device=device, save_dir=save_dir, model_id=model_id,
                      eval_every=cfg.get("eval_every", 500),
                      patience=cfg.get("patience", 15))
    best = trainer.fit(train_dl, val_dl,
                       max_steps=cfg.get("max_steps", 350_000),
                       warmup_steps=cfg.get("warmup_steps", 1000))
    print(f"Model {model_id} best auROC: {best:.4f}")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--config", default=None,
                   help="JSON config file (from tuning output)")
    p.add_argument("--save_dir", default="checkpoints/ensemble/")
    p.add_argument("--n_models", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)

    for i in range(args.n_models):
        print(f"\n{'=' * 50}")
        print(f"  Training model {i + 1} / {args.n_models}")
        print(f"{'=' * 50}")
        train_one(i, cfg, args.train_csv, args.val_csv,
                  args.save_dir, args.device)


if __name__ == "__main__":
    main()
