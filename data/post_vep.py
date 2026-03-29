"""
data/post_vep.py

Converts SAS-filtered, VEP-annotated VCFs → training CSV.

Confirmed VCF schemas (from check_columns notebooks):

  gnomAD   | 0 samples | 665 INFO fields | AC_joint_sas/AN_joint_sas/AF_joint_sas | 26 CSQ fields
  SG10K    | 1125 samp | 7   INFO fields | AC/AN/AF + AR2/DR2/IMP imputation
             | 27 CSQ fields (has CANONICAL at [23])
  IndiGen  | 0 samples | 2   INFO fields | VRT + CSQ only — NO AF                 | 26 CSQ fields
  1000G    | 489 samp  | 13  INFO fields | AC/AN/AF/SAS_AF                        | 26 CSQ fields

File naming:
  gnomAD:  gnomad_chr{N}_pure_sas_annotated_mane.vcf.gz
  SG10K:   sg10k_chr{N}_SAS_annotated_mane.vcf.gz
  IndiGen: indigen_annotated_mane.vcf.gz  (single genome-wide file)
  1000G:   1k_chr{N}_SAS_annotated_mane.vcf.gz
"""
from __future__ import annotations
import re
import json
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
from cyvcf2 import VCF

log = logging.getLogger(__name__)

# ── CSQ index tables (hardcoded from notebook inspection) ──────────────────

_CSQ_26 = {   # gnomAD, IndiGen, 1000G
    "Allele": 0, "Consequence": 1, "IMPACT": 2, "SYMBOL": 3, "Gene": 4,
    "Feature_type": 5, "Feature": 6, "BIOTYPE": 7, "EXON": 8, "INTRON": 9,
    "HGVSc": 10, "HGVSp": 11, "cDNA_position": 12, "CDS_position": 13,
    "Protein_position": 14, "Amino_acids": 15, "Codons": 16,
    "Existing_variation": 17, "DISTANCE": 18, "STRAND": 19, "FLAGS": 20,
    "SYMBOL_SOURCE": 21, "HGNC_ID": 22, "MANE": 23,
    "MANE_SELECT": 24, "MANE_PLUS_CLINICAL": 25,
}
_CSQ_27 = {   # SG10K — has CANONICAL at 23, shifts MANE fields by 1
    **{k: v for k, v in _CSQ_26.items() if v < 23},
    "CANONICAL": 23, "MANE": 24, "MANE_SELECT": 25, "MANE_PLUS_CLINICAL": 26,
}
SOURCE_CSQ = {"gnomad": _CSQ_26, "indigen": _CSQ_26, "1000g": _CSQ_26, "sg10k": _CSQ_27}


def _parse_csq(s: str, source: str) -> dict:
    idx = SOURCE_CSQ.get(source, _CSQ_26)
    p = s.split("|")
    while len(p) < len(idx):
        p.append("")
    return {f: p[i] for f, i in idx.items()}

# ── Source-aware AF extraction ─────────────────────────────────────────────


def _scalar(variant, key, cast=float):
    v = variant.INFO.get(key)
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = v[0]
    try:
        return cast(v)
    except BaseException:
        return None


def get_af(variant, source: str):
    if source == "gnomad":
        ac = _scalar(variant, "AC_joint_sas", int)
        an = _scalar(variant, "AN_joint_sas", int)
        af = _scalar(variant, "AF_joint_sas", float) or (ac / an if ac and an else None)
        return ac, an, af
    if source == "sg10k":
        return _scalar(variant, "AC", int), _scalar(variant, "AN", int), _scalar(variant, "AF", float)
    if source == "indigen":
        return None, None, None          # no AF fields — confirmed
    if source == "1000g":
        sas_af = _scalar(variant, "SAS_AF", float)
        return (_scalar(variant, "AC", int), _scalar(variant, "AN", int),
                sas_af or _scalar(variant, "AF", float))
    return None, None, None

# ── Imputation QC (SG10K only) ─────────────────────────────────────────────


MIN_AR2 = 0.3


def passes_qc(variant, source: str) -> bool:
    if source != "sg10k":
        return True
    if variant.INFO.get("IMP") is None:
        return True
    ar2 = _scalar(variant, "AR2", float)
    return ar2 is not None and ar2 >= MIN_AR2

# ── File discovery ─────────────────────────────────────────────────────────


def find_vcf(vcf_dir: Path, source: str, chrom: str) -> Optional[str]:
    n = chrom.replace("chr", "")
    patterns = {
        "gnomad": [f"gnomad_chr{n}_pure_sas_annotated_mane.vcf.gz"],
        "sg10k": [f"sg10k_chr{n}_SAS_annotated_mane.vcf.gz"],
        "indigen": ["indigen_annotated_mane.vcf.gz"],
        "1000g": [f"1k_chr{n}_SAS_annotated_mane.vcf.gz"],
    }.get(source, [])
    for name in patterns:
        p = vcf_dir / name
        if p.exists():
            return str(p)
    return None

# ── VCF parser ─────────────────────────────────────────────────────────────


_AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
        "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
        "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
        "Tyr": "Y", "Val": "V", "Ter": "*"}


def _parse_aa(amino_acids: str, hgvsp: str):
    if "/" in amino_acids:
        parts = amino_acids.strip().split("/")
        if len(parts) == 2:
            r, a = parts[0].strip(), parts[1].strip()
            if len(r) == 1 and len(a) == 1 and r.isalpha() and a.isalpha():
                return r.upper(), a.upper()
    if "p." in hgvsp:
        m = re.search(r'p\.([A-Z][a-z]{2})\d+([A-Z][a-z]{2})', hgvsp)
        if m:
            r, a = _AA3.get(m.group(1), ""), _AA3.get(m.group(2), "")
            if r and a:
                return r, a
    return "", ""


def parse_vcf(vcf_path: str, source: str,
              chromosomes: Optional[list[str]] = None) -> pd.DataFrame:
    log.info("[%-8s] Parsing %s", source, Path(vcf_path).name)
    records = []
    n_total = n_kept = 0
    for v in VCF(vcf_path):
        if chromosomes and v.CHROM not in chromosomes:
            continue
        if len(v.ALT) > 1:
            continue
        n_total += 1
        if not passes_qc(v, source):
            continue
        ac, an, af = get_af(v, source)
        raw = v.INFO.get("CSQ")
        if not raw:
            continue
        for tx_str in (raw.split(",") if isinstance(raw, str) else list(raw)):
            tx = _parse_csq(tx_str, source)
            if not tx["MANE_SELECT"].strip():
                continue
            if "missense_variant" not in tx["Consequence"]:
                continue
            if tx["BIOTYPE"] != "protein_coding":
                continue
            ref_aa, alt_aa = _parse_aa(tx["Amino_acids"], tx["HGVSp"])
            if not ref_aa or not alt_aa or ref_aa == alt_aa:
                continue
            if "*" in (ref_aa, alt_aa):
                continue
            try:
                ppos = int(tx["Protein_position"].split("-")[0])
            except BaseException:
                continue
            n_kept += 1
            records.append({
                "chrom": v.CHROM, "pos": v.POS,
                "ref_allele": v.REF, "alt_allele": v.ALT[0],
                "source": source, "gene_symbol": tx["SYMBOL"],
                "transcript_id": tx["Feature"], "mane_select": tx["MANE_SELECT"],
                "hgnc_id": tx["HGNC_ID"], "protein_pos": ppos,
                "ref_aa": ref_aa, "alt_aa": alt_aa,
                "consequence": tx["Consequence"], "impact": tx["IMPACT"],
                "hgvsp": tx["HGVSp"], "existing_var": tx["Existing_variation"],
                "ac": ac, "an": an, "af": af,
                "ar2": _scalar(v, "AR2", float),
                "is_imputed": source == "sg10k" and v.INFO.get("IMP") is not None,
            })
    log.info("[%-8s] %d total → %d MANE-select missense kept", source, n_total, n_kept)
    return pd.DataFrame(records)

# ── Deduplication ──────────────────────────────────────────────────────────


_PRIORITY = {"gnomad": 0, "sg10k": 1, "1000g": 2, "indigen": 3}


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    df = df.copy()
    df["_p"] = df["source"].map(_PRIORITY).fillna(99)
    df = (df.sort_values(["_p", "an"], ascending=[True, False], na_position="last")
            .drop_duplicates(["gene_symbol", "protein_pos", "ref_aa", "alt_aa"])
            .drop(columns="_p").reset_index(drop=True))
    log.info("Dedup: %d → %d (%d removed)", n, len(df), n - len(df))
    return df

# ── Gene → UniProt mapping ─────────────────────────────────────────────────


def build_gene_map(symbols: list[str], cache: str = "data/cache/gene_uniprot.json"):
    """
    Map HGNC gene symbols → UniProt canonical accessions via MyGene.info.

    Fixes vs original:
      - Correct endpoint: /v3/query  (not /v3/gene which is for Entrez IDs)
      - Correct body:     JSON list  (not form-encoded comma string)
      - Response shape:   {"hits": [...]}  (not bare list)
      - Hard error log if zero genes resolved (surfaces API failures immediately)
    """
    path = Path(cache)
    mapping = json.loads(path.read_text()) if path.exists() else {}
    todo = [g for g in symbols if g and g not in mapping]

    if not todo:
        log.info("Gene map: all %d genes already cached", len(mapping))
        return mapping

    log.info("MyGene.info: querying %d genes", len(todo))
    sess = requests.Session()
    resolved = failed = 0

    for start in range(0, len(todo), 1000):
        chunk = todo[start:start + 1000]
        try:
            # POST to /v3/query with JSON body — correct API for bulk symbol lookup
            r = sess.post(
                "https://mygene.info/v3/query",
                json={
                    "q": chunk,
                    "scopes": "symbol",
                    "fields": "uniprot,symbol",
                    "species": "human",
                    "size": len(chunk),
                },
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if not r.ok:
                log.warning("MyGene HTTP %d for chunk %d-%d: %s",
                            r.status_code, start, start + len(chunk), r.text[:200])
                continue

            data = r.json()
            # /v3/query returns {"hits": [...], "total": N} — not a bare list
            hits = data.get("hits", data) if isinstance(data, dict) else data

            for hit in hits:
                sym = hit.get("query", hit.get("symbol", ""))
                if not sym:
                    continue
                up = hit.get("uniprot", {}) or {}
                sp = up.get("Swiss-Prot", "")
                if isinstance(sp, list):
                    sp = sp[0] if sp else ""
                if not sp:
                    tr = up.get("TrEMBL", "")
                    sp = (tr[0] if isinstance(tr, list) else tr) or ""
                mapping[sym] = sp
                if sp:
                    resolved += 1
                else:
                    failed += 1

        except Exception as e:
            log.warning("MyGene chunk %d-%d failed: %s", start, start + len(chunk), e)
        time.sleep(0.5)

    log.info("Gene map: %d resolved, %d no UniProt ID", resolved, failed)

    if resolved == 0 and len(todo) > 0:
        log.error(
            "ZERO genes resolved from MyGene.info — API may be unreachable "
            "or response format changed. Sample genes: %s", todo[:5]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2))
    log.info("Gene map saved: %s (%d entries)", cache, len(mapping))
    return mapping

# ── UniProt sequences ──────────────────────────────────────────────────────


def fetch_sequences(ids: list[str], cache: str = "data/cache/uniprot_seqs.json"):
    path = Path(cache)
    store = json.loads(path.read_text()) if path.exists() else {}
    todo = [u for u in ids if u and u not in store]
    if todo:
        log.info("Fetching %d UniProt sequences", len(todo))
        sess = requests.Session()
        sess.headers["User-Agent"] = "esm-missense/1.0"
        for i, uid in enumerate(todo):
            for attempt in range(3):
                try:
                    r = sess.get(f"https://rest.uniprot.org/uniprotkb/{uid}.fasta", timeout=15)
                    if r.ok:
                        store[uid] = "".join(r.text.strip().split("\n")[1:])
                        break
                    elif r.status_code == 404:
                        store[uid] = None
                        break
                except BaseException:
                    pass
                time.sleep(2**attempt)
            else:
                store[uid] = None
            if (i + 1) % 200 == 0:
                path.write_text(json.dumps(store))
                log.info("  %d/%d sequences fetched", i + 1, len(todo))
        path.write_text(json.dumps(store))
    return store

# ── MAF weight ─────────────────────────────────────────────────────────────


def maf_weight(af: Optional[float], source: str) -> float:
    if source == "indigen":
        return 0.2
    if af is None or (isinstance(af, float) and np.isnan(af)):
        return 0.2
    for lo, hi, w in [(2e-4, 1e9, 1.0), (7.37e-5, 2e-4, 0.8),
                      (2.71e-5, 7.37e-5, 0.4), (1e-5, 2.71e-5, 0.2), (0, 1e-5, 0.1)]:
        if lo <= af < hi:
            return w
    return 1.0

# ── Validation ─────────────────────────────────────────────────────────────


def validate(seq, pos, ref, alt) -> bool:
    return bool(seq) and 1 <= pos <= len(seq) and seq[pos - 1] == ref and ref != alt and "*" not in (ref, alt)

# ── Main pipeline ──────────────────────────────────────────────────────────


def build_training_csv(
    annotated_dirs: dict[str, str],
    output_csv: str,
    chromosomes: Optional[list[str]] = None,
    seq_cache_path: str = "data/cache/uniprot_seqs.json",
    gene_map_cache: str = "data/cache/gene_uniprot.json",
    min_an: int = 100,
) -> pd.DataFrame:

    if chromosomes is None:
        chromosomes = [f"chr{i}" for i in range(1, 23)]

    # Parse all sources
    frames = []
    indigen_done = False
    for source, vcf_dir in annotated_dirs.items():
        vcf_dir = Path(vcf_dir)
        src_frames = []
        if source == "indigen" and not indigen_done:
            p = find_vcf(vcf_dir, "indigen", "chr1")
            if p:
                src_frames.append(parse_vcf(p, "indigen", chromosomes))
                indigen_done = True
        else:
            for chrom in chromosomes:
                p = find_vcf(vcf_dir, source, chrom)
                if not p:
                    log.warning("Not found: %s %s in %s", source, chrom, vcf_dir)
                    continue
                try:
                    src_frames.append(parse_vcf(p, source, [chrom]))
                except Exception as e:
                    log.error("%s %s: %s", source, chrom, e)
        if src_frames:
            df_src = pd.concat(src_frames, ignore_index=True)
            log.info("%-8s: %d variants", source, len(df_src))
            frames.append(df_src)

    df = pd.concat(frames, ignore_index=True)
    log.info("All sources: %d variants", len(df))

    # AN filter (skip IndiGen which has no AN)
    has_an = df["an"].notna() & (df["an"] > 0)
    df = df[~(has_an & (df["an"] < min_an))].copy()
    log.info("After AN filter: %d", len(df))

    # Dedup
    df = deduplicate(df)

    # Gene → UniProt
    gene_map = build_gene_map(df["gene_symbol"].dropna().unique().tolist(), gene_map_cache)
    df["protein_id"] = df["gene_symbol"].map(gene_map)
    df = df[df["protein_id"].notna() & (df["protein_id"] != "")].copy()
    log.info("After UniProt resolution: %d", len(df))

    # Sequences
    seq_map = fetch_sequences(df["protein_id"].unique().tolist(), seq_cache_path)
    df["sequence"] = df["protein_id"].map(seq_map)

    # Validate
    n = len(df)
    df = df[df.apply(lambda r: validate(r.get("sequence"), r["protein_pos"],
                                        r["ref_aa"], r["alt_aa"]), axis=1)].copy()
    log.info("After validation: %d (removed %d ref-mismatch)", len(df), n - len(df))

    # Labels + weights
    df["label"] = 0
    df["weight"] = df.apply(lambda r: maf_weight(r["af"], r["source"]), axis=1)

    out = df.rename(columns={"protein_pos": "position"})[[
        "protein_id", "sequence", "position", "ref_aa", "alt_aa",
        "label", "weight", "source", "gene_symbol", "transcript_id",
        "mane_select", "af", "ac", "an", "ar2", "is_imputed",
        "chrom", "pos", "hgvsp", "existing_var",
    ]].reset_index(drop=True)

    _qc(out)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    log.info("Saved: %s (%d variants)", output_csv, len(out))
    return out


def _qc(df):
    print(f"\n{'=' * 55}")
    print("  Training data QC")
    print(f"{'=' * 55}")
    print(f"  Variants  : {len(df):,}")
    print(f"  Proteins  : {df['protein_id'].nunique():,}")
    print(f"  Genes     : {df['gene_symbol'].nunique():,}")
    for src, g in df.groupby("source"):
        note = " (no AF→weight=0.2)" if src == "indigen" else ""
        print(f"  {src:<10}: {len(g):>8,}{note}")
    print(f"{'=' * 55}\n")
