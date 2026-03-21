#!/usr/bin/env python3
"""
Train a single ESM-Missense model.
"""

import sys, argparse, logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset        import SASVariantDataset, collate_variants
from model.esm_missense  import ESMMissense
from training.loss       import clipped_sigmoid_xent
from training.trainer    import Trainer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",      required=True)
    p.add_argument("--val_csv",        required=True)
    p.add_argument("--save_dir",       default="checkpoints/")
    p.add_argument("--model_id",       type=int,   default=0)
    p.add_argument("--freeze_layers",  type=int,   default=30)
    p.add_argument("--proj_dim",       type=int,   default=512)
    p.add_argument("--hidden_dim",     type=int,   default=256)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--esm_lr",         type=float, default=1e-5)
    p.add_argument("--head_lr",        type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--max_steps",      type=int,   default=350_000)
    p.add_argument("--warmup_steps",   type=int,   default=1000)
    p.add_argument("--eval_every",     type=int,   default=500)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--seed",           type=int,   default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed + args.model_id)
    device = args.device

    train_ds = SASVariantDataset(args.train_csv)
    val_ds   = SASVariantDataset(args.val_csv)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_variants,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=32,
                              shuffle=False, collate_fn=collate_variants,
                              num_workers=2, pin_memory=True)

    model = ESMMissense(
        freeze_esm_layers=args.freeze_layers,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params     = list(model.fusion.parameters()) + \
                      list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.esm_lr},
         {"params": head_params,     "lr": args.head_lr}],
        weight_decay=args.weight_decay, eps=1e-5,
    )

    trainer = Trainer(
        model=model, optimizer=optimizer,
        loss_fn=clipped_sigmoid_xent,
        device=device, save_dir=args.save_dir,
        model_id=args.model_id, eval_every=args.eval_every,
    )
    best = trainer.fit(train_loader, val_loader,
                       max_steps=args.max_steps,
                       warmup_steps=args.warmup_steps)
    print(f"Best val auROC: {best:.4f}")

if __name__ == "__main__":
    main()
