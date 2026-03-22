#!/usr/bin/env python3
"""
Run full benchmark suite on a checkpoint.
"""

from evaluation.benchmark import BenchmarkSuite
from training.trainer import EMAModel
from model.esm_missense import ESMMissense
from data.pipeline import DataPipeline
import sys
import argparse
import logging
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--benchmark_dir", required=True)
    p.add_argument("--val_csv", default=None)
    p.add_argument("--output_dir", default="reports/")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_bootstrap", type=int, default=999)
    p.add_argument("--benchmarks", nargs="+",
                   default=["clinvar_test", "clinvar_balanced",
                            "cancer_hotspot", "de_novo", "sas_test"])
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    kw = ckpt.get("model_kwargs", {})
    model = ESMMissense(**kw).to(args.device)
    if "ema_state" in ckpt:
        ema = EMAModel(model)
        ema.load_state_dict(ckpt["ema_state"])
        ema.apply_to(model)
    else:
        model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()

    pipeline = DataPipeline()
    suite = BenchmarkSuite(args.benchmark_dir, args.n_bootstrap)

    if args.val_csv:
        suite.calibrate(model, pipeline, args.val_csv,
                        args.device, args.batch_size)

    results = suite.run_all(model, pipeline, args.device,
                            args.batch_size, args.benchmarks)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir) / "benchmark_results.csv"
    results.to_csv(out, index=False)
    print(f"\n{results.to_string()}\n\nSaved: {out}")


if __name__ == "__main__":
    main()
