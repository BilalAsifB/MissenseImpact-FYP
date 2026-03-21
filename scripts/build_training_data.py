#!/usr/bin/env python3
"""
Full pipeline: annotated VCFs → split training CSVs.

Usage:
    python scripts/build_training_data.py \
        --gnomad_dir  /path/annotated_data_mane/gnomad/gnomad_pure_sas_annotated/ \
        --sg10k_dir   /path/annotated_data_mane/sg10k/sg10k_annotated/ \
        --indigen_dir /path/annotated_data_mane/indigen/indigen_annotated/ \
        --thousandg_dir /path/annotated_data_mane/1k/1k_annotated/ \
        --output_dir  data/processed/ \
        --seq_cache   data/cache/uniprot_seqs.json \
        --gene_cache  data/cache/gene_uniprot.json
"""

import sys, json, argparse, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.post_vep   import build_training_csv
from data.splits     import split_by_protein, save_splits

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gnomad_dir",    required=True)
    p.add_argument("--sg10k_dir",     required=True)
    p.add_argument("--indigen_dir",   required=True)
    p.add_argument("--thousandg_dir", required=True)
    p.add_argument("--output_dir",    default="data/processed")
    p.add_argument("--seq_cache",     default="data/cache/uniprot_seqs.json")
    p.add_argument("--gene_cache",    default="data/cache/gene_uniprot.json")
    p.add_argument("--chromosomes",   nargs="*", default=None)
    p.add_argument("--min_an",        type=int, default=100)
    p.add_argument("--val_frac",      type=float, default=0.1)
    p.add_argument("--test_frac",     type=float, default=0.1)
    args = p.parse_args()

    Path("data/cache").mkdir(parents=True, exist_ok=True)

    annotated_dirs = {
        "gnomad":  args.gnomad_dir,
        "sg10k":   args.sg10k_dir,
        "indigen": args.indigen_dir,
        "1000g":   args.thousandg_dir,
    }

    # Step 1: VCFs → single benign CSV
    raw_csv = str(Path(args.output_dir) / "benign_all.csv")
    df = build_training_csv(
        annotated_dirs  = annotated_dirs,
        output_csv      = raw_csv,
        chromosomes     = args.chromosomes,
        seq_cache_path  = args.seq_cache,
        gene_map_cache  = args.gene_cache,
        min_an          = args.min_an,
    )

    # Step 2: Position-aware splits
    train, val, test = split_by_protein(df, args.val_frac, args.test_frac)
    save_splits(train, val, test, args.output_dir)

if __name__ == "__main__":
    main()
