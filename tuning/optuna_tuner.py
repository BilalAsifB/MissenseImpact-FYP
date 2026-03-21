"""
3-phase Optuna hyperparameter search.
"""

from __future__ import annotations
import json, logging, argparse
import numpy as np
import torch
import optuna
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Phase-aware hyperparameter sampling ───────────────────────────────────

def sample(trial: optuna.Trial, phase: int) -> dict:
    h = {
        "clip_neg":    trial.suggest_float("clip_neg",  -0.5, 1.0,  step=0.25),
        "clip_pos":    trial.suggest_float("clip_pos",  -2.0, 0.0,  step=0.25),
        "proj_dim":    trial.suggest_categorical("proj_dim",  [256, 512, 768]),
        "hidden_dim":  trial.suggest_categorical("hidden_dim",[128, 256, 512]),
        "dropout":     trial.suggest_float("dropout",    0.05, 0.4),
        "head_lr":     trial.suggest_float("head_lr",    1e-5, 5e-3, log=True),
        "weight_decay":trial.suggest_float("weight_decay",1e-4,1e-1,log=True),
        "batch_size":  trial.suggest_categorical("batch_size",[8, 16, 32]),
        "grad_clip":   trial.suggest_float("grad_clip",  0.05, 1.0, log=True),
    }
    if phase == 1:
        h["freeze_esm_layers"] = 33   # fully frozen — fast head search
        h["esm_lr"] = 0.0
    elif phase == 2:
        h["freeze_esm_layers"] = trial.suggest_int("freeze_layers", 24, 33)
        h["esm_lr"] = trial.suggest_float("esm_lr", 1e-6, 1e-4, log=True)
    return h


def make_objective(train_csv, val_csv, device, phase, max_steps=2000, seed=42):
    def objective(trial):
        h = sample(trial, phase)
        from data.dataset       import SASVariantDataset, collate_variants
        from model.esm_missense import ESMMissense
        from training.loss      import clipped_sigmoid_xent

        torch.manual_seed(seed + trial.number)
        model = ESMMissense(
            freeze_esm_layers=h["freeze_esm_layers"],
            proj_dim=h["proj_dim"], hidden_dim=h["hidden_dim"],
            dropout=h["dropout"],
        ).to(device)

        train_dl = DataLoader(
            SASVariantDataset(train_csv), batch_size=h["batch_size"],
            shuffle=True, collate_fn=collate_variants, num_workers=2,
            pin_memory=True, drop_last=True)
        val_dl = DataLoader(
            SASVariantDataset(val_csv), batch_size=32, shuffle=False,
            collate_fn=collate_variants, num_workers=2)

        bb = [p for p in model.backbone.parameters() if p.requires_grad]
        hd = list(model.fusion.parameters())+list(model.classifier.parameters())
        opt = torch.optim.AdamW(
            [{"params": bb, "lr": h["esm_lr"]},
             {"params": hd, "lr": h["head_lr"]}],
            weight_decay=h["weight_decay"], eps=1e-5)

        step = 0
        model.train()
        for epoch in range(100):
            for batch in train_dl:
                if step >= max_steps: break
                batch = {k: v.to(device) if isinstance(v,torch.Tensor) else v
                         for k,v in batch.items()}
                out  = model(batch)
                loss = clipped_sigmoid_xent(
                    out["logit"], batch["labels"],
                    clip_neg=h["clip_neg"], clip_pos=h["clip_pos"],
                    weights=batch.get("weights")).mean()
                if not torch.isfinite(loss): raise optuna.TrialPruned()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), h["grad_clip"])
                opt.step(); step += 1
            if step >= max_steps: break

            # Intermediate reporting for pruning
            auroc = _val_auroc(model, val_dl, device)
            trial.report(auroc, step)
            if trial.should_prune(): raise optuna.TrialPruned()

        return _val_auroc(model, val_dl, device)
    return objective


@torch.no_grad()
def _val_auroc(model, loader, device):
    model.eval()
    lg, lb = [], []
    for b in loader:
        b = {k: v.to(device) if isinstance(v,torch.Tensor) else v
             for k,v in b.items()}
        lg.append(model(b)["logit"].cpu().numpy())
        lb.append(b["labels"].cpu().numpy())
    model.train()
    import numpy as np
    lg, lb = np.concatenate(lg), np.concatenate(lb)
    return float(roc_auc_score(lb,lg)) if len(np.unique(lb))>1 else 0.5


def run_study(train_csv, val_csv, n_trials=50, phase=1,
              max_steps=2000, device="cuda", storage=None,
              study_name=None, seed=42) -> optuna.Study:
    study = optuna.create_study(
        study_name=study_name or f"esm_missense_p{phase}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=max(10,n_trials//5), seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=500),
        storage=storage, load_if_exists=True,
    )
    study.optimize(
        make_objective(train_csv, val_csv, device, phase, max_steps, seed),
        n_trials=n_trials, show_progress_bar=True,
        catch=(RuntimeError, torch.cuda.OutOfMemoryError),
    )
    best = study.best_trial
    print(f"\nBest auROC: {best.value:.4f}")
    for k,v in best.params.items(): print(f"  {k:<28}: {v}")
    return study


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s  %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",  required=True)
    p.add_argument("--val_csv",    required=True)
    p.add_argument("--phase",      type=int, default=1, choices=[1,2])
    p.add_argument("--n_trials",   type=int, default=50)
    p.add_argument("--max_steps",  type=int, default=2000)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--storage",    default=None)
    p.add_argument("--output_json",default="best_hparams.json")
    args = p.parse_args()

    study = run_study(args.train_csv, args.val_csv, args.n_trials,
                      args.phase, args.max_steps, args.device, args.storage)
    with open(args.output_json,"w") as f:
        json.dump(dict(study.best_params), f, indent=2)
    print(f"\nSaved: {args.output_json}")
