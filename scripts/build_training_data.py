#!/usr/bin/env python3
"""
scripts/build_training_data.py

Full pipeline: annotated VCFs → split training CSVs.

Usage:
    python scripts/build_training_data.py \
        --gnomad_dir    /path/gnomad_pure_sas_annotated/ \
        --sg10k_dir     /path/sg10k_annotated/ \
        --indigen_dir   /path/indigen_annotated/ \
        --thousandg_dir /path/1k_annotated/ \
        --output_dir    data/processed/ \
        --seq_cache     data/cache/uniprot_seqs.json \
        --gene_cache    data/cache/gene_uniprot.json \
        --checkpoint_dir data/cache/vcf_checkpoints

Checkpointing:
    Each (source, chromosome) pair is saved to a parquet file in
    --checkpoint_dir immediately after parsing. If the run is interrupted,
    re-running the script resumes from the last completed chromosome instead
    of re-parsing from scratch. Delete --checkpoint_dir to force a full re-parse.
"""

from data.splits import split_by_protein, save_splits
from data.post_vep import build_training_csv
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s  %(message)s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gnomad_dir", required=True)
    p.add_argument("--sg10k_dir", required=True)
    p.add_argument("--indigen_dir", required=True)
    p.add_argument("--thousandg_dir", required=True)
    p.add_argument("--output_dir", default="data/processed")
    p.add_argument("--seq_cache", default="data/cache/uniprot_seqs.json")
    p.add_argument("--gene_cache", default="data/cache/gene_uniprot.json")
    p.add_argument("--checkpoint_dir", default="data/cache/vcf_checkpoints",
                   help="Directory for per-(source, chrom) parquet checkpoints")
    p.add_argument("--chromosomes", nargs="*", default=None)
    p.add_argument("--min_an", type=int, default=100)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    args = p.parse_args()

    Path("data/cache").mkdir(parents=True, exist_ok=True)

    annotated_dirs = {
        "gnomad": args.gnomad_dir,
        "sg10k": args.sg10k_dir,
        "indigen": args.indigen_dir,
        "1000g": args.thousandg_dir,
    }

    # Step 1: VCFs → single benign CSV (with per-chromosome checkpointing)
    raw_csv = str(Path(args.output_dir) / "benign_all.csv")
    df = build_training_csv(
        annotated_dirs=annotated_dirs,
        output_csv=raw_csv,
        chromosomes=args.chromosomes,
        seq_cache_path=args.seq_cache,
        gene_map_cache=args.gene_cache,
        min_an=args.min_an,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Step 2: Position-aware splits
    train, val, test = split_by_protein(df, args.val_frac, args.test_frac)
    save_splits(train, val, test, args.output_dir)


if __name__ == "__main__":
    main()
