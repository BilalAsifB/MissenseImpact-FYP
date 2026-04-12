"""
Unit tests for data/post_vep.py helper logic and checkpoint behavior.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data import post_vep


class _Variant:
    def __init__(self, info: dict):
        self.INFO = info


class _VRecord:
    def __init__(self, chrom: str, pos: int, ref: str, alt: list[str], info: dict):
        self.CHROM = chrom
        self.POS = pos
        self.REF = ref
        self.ALT = alt
        self.INFO = info


def _make_csq(
    source: str = "gnomad",
    consequence: str = "missense_variant",
    biotype: str = "protein_coding",
    mane_select: str = "NM_000001.1",
    amino_acids: str = "A/V",
    hgvsp: str = "ENSP000:p.Ala10Val",
    protein_position: str = "10",
):
    idx = post_vep.SOURCE_CSQ[source]
    fields = [""] * len(idx)
    fields[idx["Consequence"]] = consequence
    fields[idx["SYMBOL"]] = "GENE1"
    fields[idx["Feature"]] = "ENST000001"
    fields[idx["BIOTYPE"]] = biotype
    fields[idx["HGNC_ID"]] = "HGNC:1"
    fields[idx["MANE_SELECT"]] = mane_select
    fields[idx["Protein_position"]] = protein_position
    fields[idx["Amino_acids"]] = amino_acids
    fields[idx["HGVSp"]] = hgvsp
    fields[idx["IMPACT"]] = "MODERATE"
    fields[idx["Existing_variation"]] = "rs1"
    return "|".join(fields)


def test_scalar_handles_list_values_and_cast():
    v = _Variant({"AF": [0.25]})
    assert post_vep._scalar(v, "AF", float) == 0.25


def test_scalar_returns_none_on_missing_or_bad_cast():
    v = _Variant({"AF": "not_a_float"})
    assert post_vep._scalar(v, "MISSING", float) is None
    assert post_vep._scalar(v, "AF", float) is None


def test_get_af_gnomad_prefers_precomputed_af():
    v = _Variant({"AC_joint_sas": 10, "AN_joint_sas": 100, "AF_joint_sas": 0.12})
    ac, an, af = post_vep.get_af(v, "gnomad")
    assert (ac, an, af) == (10, 100, 0.12)


def test_get_af_gnomad_falls_back_to_ac_over_an():
    v = _Variant({"AC_joint_sas": 10, "AN_joint_sas": 100})
    _, _, af = post_vep.get_af(v, "gnomad")
    assert af == 0.1


def test_get_af_1000g_prefers_sas_af_over_af():
    v = _Variant({"AC": 3, "AN": 200, "AF": 0.02, "SAS_AF": 0.03})
    _, _, af = post_vep.get_af(v, "1000g")
    assert af == 0.03


def test_get_af_indigen_returns_missing():
    v = _Variant({})
    assert post_vep.get_af(v, "indigen") == (None, None, None)


def test_passes_qc_for_non_sg10k_always_true():
    assert post_vep.passes_qc(_Variant({"IMP": True, "AR2": 0.0}), "gnomad")


def test_passes_qc_sg10k_with_no_imp_is_true():
    assert post_vep.passes_qc(_Variant({"AR2": 0.0}), "sg10k")


def test_passes_qc_sg10k_requires_ar2_when_imputed():
    assert not post_vep.passes_qc(_Variant({"IMP": True}), "sg10k")
    assert not post_vep.passes_qc(_Variant({"IMP": True, "AR2": 0.29}), "sg10k")
    assert post_vep.passes_qc(_Variant({"IMP": True, "AR2": post_vep.MIN_AR2}), "sg10k")


def test_find_vcf_matches_expected_filename(tmp_path):
    p = tmp_path / "gnomad_chr1_pure_sas_annotated_mane.vcf.gz"
    p.write_text("stub")
    assert post_vep.find_vcf(tmp_path, "gnomad", "chr1") == str(p)


def test_find_vcf_returns_none_when_missing(tmp_path):
    assert post_vep.find_vcf(tmp_path, "sg10k", "chr1") is None


def test_parse_aa_from_one_letter_field():
    assert post_vep._parse_aa("A/V", "") == ("A", "V")


def test_parse_aa_from_hgvsp_three_letter_code():
    assert post_vep._parse_aa("", "ENSP:p.Ala123Val") == ("A", "V")


def test_parse_aa_invalid_returns_empty():
    assert post_vep._parse_aa("bad", "bad") == ("", "")


def test_parse_csq_uses_source_specific_schema():
    # SG10K has 27 fields with CANONICAL at index 23 and MANE_SELECT at 25.
    fields = [f"v{i}" for i in range(27)]
    parsed = post_vep._parse_csq("|".join(fields), "sg10k")
    assert parsed["CANONICAL"] == "v23"
    assert parsed["MANE_SELECT"] == "v25"


def test_parse_csq_pads_missing_fields():
    parsed = post_vep._parse_csq("a|b|c", "gnomad")
    assert parsed["Allele"] == "a"
    assert parsed["MANE_SELECT"] == ""


def test_maf_weight_bins_and_defaults():
    assert post_vep.maf_weight(None, "gnomad") == 0.2
    assert post_vep.maf_weight(1e-6, "gnomad") == 0.1
    assert post_vep.maf_weight(1.5e-5, "gnomad") == 0.2
    assert post_vep.maf_weight(3e-5, "gnomad") == 0.4
    assert post_vep.maf_weight(1e-4, "gnomad") == 0.8
    assert post_vep.maf_weight(3e-4, "gnomad") == 1.0
    assert post_vep.maf_weight(0.2, "indigen") == 0.2


def test_validate_checks_bounds_ref_and_non_synonymous():
    seq = "ACDE"
    assert post_vep.validate(seq, 1, "A", "V")
    assert not post_vep.validate("", 1, "A", "V")
    assert not post_vep.validate(seq, 0, "A", "V")
    assert not post_vep.validate(seq, 1, "C", "V")
    assert not post_vep.validate(seq, 1, "A", "A")
    assert not post_vep.validate(seq, 1, "A", "*")


def test_deduplicate_uses_source_priority_then_highest_an():
    df = pd.DataFrame(
        [
            {"source": "sg10k", "an": 150, "gene_symbol": "GENE1", "protein_pos": 10, "ref_aa": "A", "alt_aa": "V"},
            {"source": "gnomad", "an": 50, "gene_symbol": "GENE1", "protein_pos": 10, "ref_aa": "A", "alt_aa": "V"},
            {"source": "1000g", "an": 300, "gene_symbol": "GENE2", "protein_pos": 8, "ref_aa": "G", "alt_aa": "D"},
            {"source": "1000g", "an": 100, "gene_symbol": "GENE2", "protein_pos": 8, "ref_aa": "G", "alt_aa": "D"},
        ]
    )
    out = post_vep.deduplicate(df)
    assert len(out) == 2
    g1 = out[out["gene_symbol"] == "GENE1"].iloc[0]
    assert g1["source"] == "gnomad"
    g2 = out[out["gene_symbol"] == "GENE2"].iloc[0]
    assert g2["an"] == 300


def test_ckpt_path_format(tmp_path):
    ckpt = post_vep._ckpt_path(tmp_path, "sg10k", "chr5")
    assert ckpt.name == "sg10k_chr5.parquet"


def test_load_or_parse_loads_existing_checkpoint(monkeypatch, tmp_path):
    ckpt = post_vep._ckpt_path(tmp_path, "gnomad", "chr1")
    expected = pd.DataFrame([{"a": 1}])
    ckpt.write_text("stub")

    called = {"parse": False}

    def _fake_parse(*args, **kwargs):
        called["parse"] = True
        return pd.DataFrame([{"a": 999}])

    monkeypatch.setattr(post_vep.pd, "read_parquet", lambda p: expected)
    monkeypatch.setattr(post_vep, "parse_vcf", _fake_parse)

    out = post_vep._load_or_parse("x.vcf.gz", "gnomad", "chr1", ["chr1"], tmp_path)
    assert out.equals(expected)
    assert not called["parse"]


def test_load_or_parse_parses_and_writes_checkpoint(monkeypatch, tmp_path):
    expected = pd.DataFrame([{"chrom": "chr1", "pos": 123}])
    written = {"path": None}

    def _fake_parse(vcf_path, source, chromosomes):
        assert vcf_path == "x.vcf.gz"
        assert source == "gnomad"
        assert chromosomes == ["chr1"]
        return expected

    def _fake_to_parquet(self, path, index=False):
        written["path"] = Path(path)
        Path(path).write_text("ckpt")

    monkeypatch.setattr(post_vep, "parse_vcf", _fake_parse)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)

    out = post_vep._load_or_parse("x.vcf.gz", "gnomad", "chr1", ["chr1"], tmp_path)
    ckpt = post_vep._ckpt_path(tmp_path, "gnomad", "chr1")

    assert written["path"] == ckpt
    assert ckpt.exists()
    assert out.equals(expected)


def test_qc_prints_expected_sections(capsys):
    df = pd.DataFrame(
        [
            {"protein_id": "P1", "gene_symbol": "G1", "source": "gnomad"},
            {"protein_id": "P2", "gene_symbol": "G2", "source": "indigen"},
        ]
    )
    post_vep._qc(df)
    out = capsys.readouterr().out
    assert "Training data QC" in out
    assert "gnomad" in out
    assert "indigen" in out


def test_parse_vcf_keeps_only_valid_mane_missense_records(monkeypatch):
    good = _VRecord(
        chrom="chr1",
        pos=101,
        ref="A",
        alt=["T"],
        info={"CSQ": _make_csq("gnomad"), "AC_joint_sas": 1, "AN_joint_sas": 10, "AF_joint_sas": 0.1},
    )
    non_missense = _VRecord(
        chrom="chr1",
        pos=102,
        ref="A",
        alt=["T"],
        info={"CSQ": _make_csq("gnomad", consequence="synonymous_variant")},
    )
    no_mane = _VRecord(
        chrom="chr1",
        pos=103,
        ref="A",
        alt=["T"],
        info={"CSQ": _make_csq("gnomad", mane_select="")},
    )
    not_protein_coding = _VRecord(
        chrom="chr1",
        pos=104,
        ref="A",
        alt=["T"],
        info={"CSQ": _make_csq("gnomad", biotype="lncRNA")},
    )
    multi_allelic = _VRecord(
        chrom="chr1",
        pos=105,
        ref="A",
        alt=["T", "G"],
        info={"CSQ": _make_csq("gnomad")},
    )

    monkeypatch.setattr(post_vep, "VCF", lambda _: [good, non_missense, no_mane, not_protein_coding, multi_allelic])
    out = post_vep.parse_vcf("fake.vcf.gz", "gnomad", chromosomes=["chr1"])

    assert len(out) == 1
    row = out.iloc[0]
    assert row["chrom"] == "chr1"
    assert row["pos"] == 101
    assert row["gene_symbol"] == "GENE1"
    assert row["protein_pos"] == 10
    assert row["ref_aa"] == "A"
    assert row["alt_aa"] == "V"
    assert row["af"] == 0.1


def test_parse_vcf_applies_chromosome_filter(monkeypatch):
    v1 = _VRecord("chr1", 10, "A", ["T"], {"CSQ": _make_csq("gnomad")})
    v2 = _VRecord("chr2", 20, "A", ["T"], {"CSQ": _make_csq("gnomad")})
    monkeypatch.setattr(post_vep, "VCF", lambda _: [v1, v2])
    out = post_vep.parse_vcf("fake.vcf.gz", "gnomad", chromosomes=["chr2"])
    assert len(out) == 1
    assert out.iloc[0]["chrom"] == "chr2"


def test_build_training_csv_runs_all_steps_with_mocks(monkeypatch, tmp_path):
    columns = [
        "chrom",
        "pos",
        "ref_allele",
        "alt_allele",
        "source",
        "gene_symbol",
        "transcript_id",
        "mane_select",
        "hgnc_id",
        "protein_pos",
        "ref_aa",
        "alt_aa",
        "consequence",
        "impact",
        "hgvsp",
        "existing_var",
        "ac",
        "an",
        "af",
        "ar2",
        "is_imputed",
    ]
    gnomad_df = pd.DataFrame(
        [
            ["chr1", 1, "A", "T", "gnomad", "G1", "ENST1", "MANE1", "HGNC:1", 1, "A", "V", "missense_variant", "MODERATE", "p.Ala1Val", "rs1", 3, 90, 0.001, None, False],
            ["chr1", 2, "A", "T", "gnomad", "G2", "ENST2", "MANE2", "HGNC:2", 2, "C", "D", "missense_variant", "MODERATE", "p.Cys2Asp", "rs2", 3, 120, 0.00005, None, False],
        ],
        columns=columns,
    )
    indigen_df = pd.DataFrame(
        [
            ["chr1", 3, "A", "T", "indigen", "G3", "ENST3", "MANE3", "HGNC:3", 1, "M", "I", "missense_variant", "MODERATE", "p.Met1Ile", "rs3", None, None, None, None, False],
        ],
        columns=columns,
    )

    def _fake_find_vcf(vcf_dir, source, chrom):
        if source == "gnomad" and chrom == "chr1":
            return "gnomad_chr1.vcf.gz"
        if source == "indigen":
            return "indigen.vcf.gz"
        return None

    def _fake_load_or_parse(vcf_path, source, chrom, chromosomes, checkpoint_dir):
        if source == "gnomad":
            return gnomad_df.copy()
        return indigen_df.copy()

    monkeypatch.setattr(post_vep, "find_vcf", _fake_find_vcf)
    monkeypatch.setattr(post_vep, "_load_or_parse", _fake_load_or_parse)
    monkeypatch.setattr(post_vep, "build_gene_map", lambda symbols, cache: {"G2": "P_G2", "G3": "P_G3"})
    monkeypatch.setattr(post_vep, "fetch_sequences", lambda ids, cache: {"P_G2": "ACDE", "P_G3": "MQQQ"})
    monkeypatch.setattr(post_vep, "_qc", lambda df: None)

    output_csv = tmp_path / "processed" / "benign_all.csv"
    out = post_vep.build_training_csv(
        annotated_dirs={"gnomad": str(tmp_path / "gn"), "indigen": str(tmp_path / "in")},
        output_csv=str(output_csv),
        chromosomes=["chr1"],
        gene_map_cache=str(tmp_path / "gene.json"),
        seq_cache_path=str(tmp_path / "seq.json"),
        checkpoint_dir=str(tmp_path / "ckpt"),
        min_an=100,
    )

    assert output_csv.exists()
    # gnomad row with AN=90 is removed by min_an filter; two rows remain.
    assert len(out) == 2
    assert set(out["protein_id"]) == {"P_G2", "P_G3"}
    assert set(out["label"]) == {0}
    # G2 has AF in 2.71e-5..7.37e-5 -> weight 0.4; indigen always 0.2
    by_gene = dict(zip(out["gene_symbol"], out["weight"]))
    assert by_gene["G2"] == 0.4
    assert by_gene["G3"] == 0.2


def test_load_or_parse_uses_all_chromosomes_for_indigen_checkpoint(monkeypatch, tmp_path):
    expected = pd.DataFrame([{"chrom": "chr1"}])
    called = {"chromosomes": None}

    def _fake_parse(vcf_path, source, chromosomes):
        called["chromosomes"] = chromosomes
        return expected

    monkeypatch.setattr(post_vep, "parse_vcf", _fake_parse)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, path, index=False: Path(path).write_text("ckpt"))

    out = post_vep._load_or_parse("indigen.vcf.gz", "indigen", "all", ["chr1", "chr2"], tmp_path)
    assert out.equals(expected)
    assert called["chromosomes"] == ["chr1", "chr2"]


def test_build_gene_map_reads_existing_cache_without_http(monkeypatch, tmp_path):
    cache = tmp_path / "gene_uniprot.json"
    cache.write_text('{"GENE1": "P12345"}')

    class _FakeSession:
        def post(self, *args, **kwargs):
            raise AssertionError("HTTP should not be called when cache already covers symbols")

    monkeypatch.setattr(post_vep.requests, "Session", lambda: _FakeSession())
    out = post_vep.build_gene_map(["GENE1"], cache=str(cache))
    assert out == {"GENE1": "P12345"}


def test_build_gene_map_queries_api_and_persists(monkeypatch, tmp_path):
    cache = tmp_path / "gene_uniprot.json"
    posted = {"chunks": []}

    class _Resp:
        ok = True
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "hits": [
                    {"query": "GENE1", "uniprot": {"Swiss-Prot": "P11111"}},
                    {"query": "GENE2", "uniprot": {"TrEMBL": ["Q22222"]}},
                    {"query": "GENE3", "uniprot": {}},
                ]
            }

    class _FakeSession:
        def post(self, url, json, headers, timeout):
            posted["chunks"].append(json["q"])
            return _Resp()

    monkeypatch.setattr(post_vep.requests, "Session", lambda: _FakeSession())
    monkeypatch.setattr(post_vep.time, "sleep", lambda _s: None)

    out = post_vep.build_gene_map(["GENE1", "GENE2", "GENE3"], cache=str(cache))
    assert out["GENE1"] == "P11111"
    assert out["GENE2"] == "Q22222"
    assert out["GENE3"] == ""
    assert posted["chunks"] == [["GENE1", "GENE2", "GENE3"]]

    persisted = post_vep.json.loads(cache.read_text())
    assert persisted["GENE1"] == "P11111"


def test_fetch_sequences_uses_cache_and_fetches_missing(monkeypatch, tmp_path):
    cache = tmp_path / "seqs.json"
    cache.write_text('{"P_CACHED": "ACDE"}')

    class _Resp:
        def __init__(self, ok, status_code, text):
            self.ok = ok
            self.status_code = status_code
            self.text = text

    class _FakeSession:
        headers = {}

        def get(self, url, timeout):
            if "P_NEW" in url:
                return _Resp(True, 200, ">sp|P_NEW|\nMKT")
            if "P_404" in url:
                return _Resp(False, 404, "")
            raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(post_vep.requests, "Session", lambda: _FakeSession())
    monkeypatch.setattr(post_vep.time, "sleep", lambda _s: None)

    out = post_vep.fetch_sequences(["P_CACHED", "P_NEW", "P_404"], cache=str(cache))
    assert out["P_CACHED"] == "ACDE"
    assert out["P_NEW"] == "MKT"
    assert out["P_404"] is None

    persisted = post_vep.json.loads(cache.read_text())
    assert persisted["P_NEW"] == "MKT"
    assert persisted["P_404"] is None
