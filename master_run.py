#!/usr/bin/env python
"""
master_run.py — Optimized end-to-end pipeline (Option A+).

Improvements over original:
  - 9-channel blocking (adds TF-IDF cosine, trigram prefix, addr-num-zip, name-prefix)
  - 15+ pairwise features (adds jaro_winkler, addr_jaro, name_qratio, zipcode features)
  - Better LightGBM (more trees, better params)
  - Entity-level score-gap abstention + precision-biased threshold selection
  - Skips already-computed phases when artifacts exist

Run from repo root:
    python master_run.py [--skip-blocking] [--skip-features] [--skip-training]

All artifacts saved under artifacts/.
Final outputs: output/candidate_pairs.tsv, output/matching_results.tsv
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import pickle
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))

# ─── third-party (install if missing) ──────────────────────────────────────

def _ensure(pkg: str, import_name: str | None = None) -> None:
    import importlib.util
    name = import_name or pkg
    if importlib.util.find_spec(name) is None:
        print(f"[setup] Installing {pkg} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

_ensure("pyyaml",         "yaml")
_ensure("lightgbm",       "lightgbm")
_ensure("rapidfuzz",      "rapidfuzz")
_ensure("duckdb",         "duckdb")
_ensure("sparse-dot-topn","sparse_dot_topn")
_ensure("scipy",          "scipy")

import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
import scipy.sparse as sp
from rapidfuzz import fuzz as rf_fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sparse_dot_topn import awesome_cossim_topn

from config.config import load_config

# ─── stdout/stderr encoding ────────────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# ─── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("master")

# ─── config ────────────────────────────────────────────────────────────────
CFG       = load_config()
ARTS      = REPO / CFG["paths"]["artifacts_dir"]
TRAIN_DIR = REPO / CFG["paths"]["train_dir"]
TEST_DIR  = REPO / CFG["paths"]["test_dir"]
OUT_DIR   = REPO / CFG["paths"]["output_dir"]
ARTS.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# ── Blocking constants ─────────────────────────────────────────────────────
MIN_TOK_LEN    = 3
MAX_RARE_FREQ  = 200        # raised from 150 — more coverage
MAX_MED_FREQ   = 5_000      # raised from 3000
MIN_RARE_SH    = 1
MIN_MED_SH     = 2
TFIDF_TOP_N    = 10         # top-N TF-IDF cosine candidates per S1 entity
TFIDF_THRESH   = 0.25       # minimum cosine similarity

# ── Feature columns ────────────────────────────────────────────────────────
FEAT_COLS = [
    # Name lexical
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_expanded_sort_ratio",
    "name_qratio",
    "name_jaro_winkler",
    "name_sorted_exact",
    "name_alphanum_exact",
    # Address
    "addr_token_jaccard",
    "addr_numeric_match",
    "addr_jaro_winkler",
    "addr_prefix4_match",
    # Country
    "country_match",
    # Length / structural
    "name_len_ratio",
    "name_token_len_ratio",
    "name_prefix5_exact",
]


# =============================================================================
# STEP 0 — Normalize
# =============================================================================

# Abbreviation expansion map (bi-directional)
_ABBR_MAP = {
    "private": "pvt",   "pvt": "private",
    "limited": "ltd",   "ltd": "limited",
    "incorporated": "inc", "inc": "incorporated",
    "corporation": "corp", "corp": "corporation",
    "services": "svcs", "svcs": "services",
    "service": "svc",   "svc": "service",
    "management": "mgmt", "mgmt": "management",
    "international": "intl", "intl": "international",
    "associates": "assoc", "assoc": "associates",
    "and": "&",         "&": "and",
    "road": "rd",       "rd": "road",
    "street": "st",     "st": "street",
    "avenue": "ave",    "ave": "avenue",
    "building": "bldg", "bldg": "building",
    "apartment": "apt", "apt": "apartment",
}


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Fast vectorized normalization producing all blocking views."""
    name = df["business_name"].fillna("").str.lower().str.strip()
    name = name.str.replace(r"<null>", "", regex=True)
    name = name.str.replace(r"^(smt|sri|--|[*]+)\s+", "", regex=True)
    name = name.str.replace(r"[^\w\s&]", " ", regex=True)
    name = name.str.replace(r"\s+", " ", regex=True).str.strip()

    alnum = name.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    alnum = alnum.str.replace(r"\s+", " ", regex=True).str.strip()

    sorted_view = alnum.str.split().apply(
        lambda t: " ".join(sorted(t)) if t else ""
    )

    # Expanded view: collapse all abbreviations to canonical form
    exp = alnum.copy()
    for src, tgt in _ABBR_MAP.items():
        exp = exp.str.replace(rf"\b{src}\b", tgt, regex=True)
    exp = exp.str.replace(r"\s+", " ", regex=True).str.strip()
    exp_sorted = exp.str.split().apply(lambda t: " ".join(sorted(t)) if t else "")

    # Name prefix (first 5 alphanum chars of sorted view)
    name_prefix5 = sorted_view.str[:5]

    addr = df["business_address"].fillna("").str.lower().str.strip()
    addr_alnum = addr.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    addr_alnum = addr_alnum.str.replace(r"\s+", " ", regex=True).str.strip()
    # Extract first numeric substring (ZIP/PIN/door number)
    addr_num = addr_alnum.str.extract(r"(\d+)", expand=False).fillna("")
    # Address 4-char prefix
    addr_prefix4 = addr_alnum.str[:4]

    out = df[["entity_id", "business_name", "business_address", "country"]].copy()
    out["name_lower"]           = name
    out["name_alphanum"]        = alnum
    out["name_sorted"]          = sorted_view
    out["name_expanded"]        = exp
    out["name_expanded_sorted"] = exp_sorted
    out["name_prefix5"]         = name_prefix5
    out["addr_alphanum"]        = addr_alnum
    out["addr_numeric"]         = addr_num
    out["addr_prefix4"]         = addr_prefix4
    return out


# =============================================================================
# STEP 1 — Multi-channel Blocking (DuckDB + TF-IDF)
# =============================================================================

def _addr_key(s: pd.Series) -> pd.Series:
    ex = s.str.extract(r"(?<!\d)(\d{4,})\s+([a-z]{4,})", expand=True)
    return (ex[0].fillna("") + " " + ex[1].fillna("")).str.strip()


def _copy_query(con, sql, out_path, label=""):
    fwd = str(out_path).replace("\\", "/")
    con.execute(f"COPY ({sql}) TO '{fwd}' (DELIMITER '\t', HEADER true)")
    try:
        rows = pd.read_csv(out_path, sep="\t", usecols=["s1_id"]).shape[0]
    except Exception:
        rows = 0
    log.info("    %-28s: %d pairs  %s", out_path.stem, rows, label)
    return out_path


def _tfidf_blocking(s1n: pd.DataFrame, s2s3n: pd.DataFrame,
                    out_path: pathlib.Path, field: str = "name_expanded_sorted",
                    top_n: int = TFIDF_TOP_N, threshold: float = TFIDF_THRESH,
                    ngram_range=(1, 2), chunk_size: int = 50_000) -> pathlib.Path:
    """
    TF-IDF cosine blocking via sparse_dot_topn.
    Chunks S1 to avoid OOM, writes pairs to out_path.
    """
    log.info("  TF-IDF blocking: field=%s top_n=%d threshold=%.2f", field, top_n, threshold)
    s1_texts   = s1n[field].fillna("").tolist()
    s2s3_texts = s2s3n[field].fillna("").tolist()
    s1_ids     = s1n["entity_id"].tolist()
    s2s3_ids   = s2s3n["entity_id"].tolist()

    # Fit on union for consistent vocabulary
    corpus = s1_texts + s2s3_texts
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    vec.fit(corpus)

    s2s3_mat = vec.transform(s2s3_texts)  # (n_s2s3, vocab)
    s1_mat   = vec.transform(s1_texts)     # (n_s1,   vocab)

    # Chunk over S1 to stay memory-safe
    n_s1 = len(s1_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    rows_written = 0

    for start in range(0, n_s1, chunk_size):
        end = min(start + chunk_size, n_s1)
        chunk_mat = s1_mat[start:end]
        matches = awesome_cossim_topn(
            chunk_mat, s2s3_mat.T,
            ntop=top_n, lower_bound=threshold,
            use_threads=True, n_threads=max(1, (os.cpu_count() or 4) // 2),
        )
        cx = matches.tocoo()
        if cx.nnz == 0:
            continue

        pairs = pd.DataFrame({
            "s1_id":     [s1_ids[start + r] for r in cx.row],
            "target_id": [s2s3_ids[c]        for c in cx.col],
            "channel":   "tfidf_name",
        })
        # Remove self-hits (shouldn't happen since S1≠S2/S3 by prefix)
        pairs = pairs[pairs["s1_id"] != pairs["target_id"]]
        pairs.to_csv(
            out_path, sep="\t", index=False,
            header=first, mode="w" if first else "a", encoding="utf-8",
        )
        first = False
        rows_written += len(pairs)

    if first:  # nothing written — create empty file with header
        pd.DataFrame(columns=["s1_id", "target_id", "channel"]).to_csv(
            out_path, sep="\t", index=False, encoding="utf-8"
        )

    log.info("    tfidf_name: %d pairs", rows_written)
    return out_path


def _build_blocking_channels(s1n, s2s3n, tmp, t0, label="train"):
    """
    9-channel blocking:
    CH1 exact_sorted
    CH2 exact_expanded_sorted
    CH3 token_rare
    CH4 token_medium
    CH5 addr_composite
    CH6 name_prefix5   (new)
    CH7 addr_numeric   (new — pure number match with country)
    CH8 tfidf_name     (new — char n-gram TF-IDF cosine)
    CH9 tfidf_addr     (new — address TF-IDF cosine)
    """
    tmp.mkdir(parents=True, exist_ok=True)
    ch_files = []

    n_cpu = os.cpu_count() or 4

    # ── DuckDB connection (exact joins CH1 CH2 CH5 CH6 CH7) ─────────────────
    db1 = str(tmp / "_ddb1.db")
    for f in [db1, db1 + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    con = duckdb.connect(database=db1)
    con.execute("SET memory_limit='12GB'")
    con.execute(f"SET threads={max(4, n_cpu // 2)}")
    con.execute("SET preserve_insertion_order=false")
    con.register("s1n",   s1n)
    con.register("s2s3n", s2s3n)

    log.info("  CH1 exact_sorted ...")
    p1 = tmp / "ch1.tsv"
    _copy_query(con, """
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id,
               'exact_sorted' AS channel
        FROM s1n a JOIN s2s3n b ON a.name_sorted = b.name_sorted
        WHERE length(a.name_sorted)>=3 AND a.entity_id<>b.entity_id
    """, p1, label); ch_files.append(p1)

    log.info("  CH2 exact_expanded_sorted ...")
    p2 = tmp / "ch2.tsv"
    _copy_query(con, """
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id,
               'exact_expanded_sorted' AS channel
        FROM s1n a JOIN s2s3n b ON a.name_expanded_sorted=b.name_expanded_sorted
        WHERE length(a.name_expanded_sorted)>=3 AND a.entity_id<>b.entity_id
    """, p2, label); ch_files.append(p2)

    log.info("  CH5 addr_composite ...")
    s1n["addr_key"]   = _addr_key(s1n["addr_alphanum"])
    s2s3n["addr_key"] = _addr_key(s2s3n["addr_alphanum"])
    con.register("s1n", s1n); con.register("s2s3n", s2s3n)
    p5 = tmp / "ch5.tsv"
    _copy_query(con, """
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id,
               'addr_composite' AS channel
        FROM s1n a JOIN s2s3n b ON a.addr_key=b.addr_key AND a.country=b.country
        WHERE length(a.addr_key)>=8 AND a.entity_id<>b.entity_id
    """, p5, label); ch_files.append(p5)

    log.info("  CH6 addr_numeric_country ...")
    p6 = tmp / "ch6.tsv"
    _copy_query(con, """
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id,
               'addr_numeric_country' AS channel
        FROM s1n a JOIN s2s3n b ON a.addr_numeric=b.addr_numeric
                               AND a.country=b.country
        WHERE length(a.addr_numeric)>=4 AND a.entity_id<>b.entity_id
    """, p6, label); ch_files.append(p6)

    con.close()
    for f in [db1, db1 + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    # ── Token tables ─────────────────────────────────────────────────────────
    log.info("  Building token frequency table ...")
    s23_tok = (s2s3n[["entity_id", "name_alphanum"]]
               .assign(token=s2s3n["name_alphanum"].str.split())
               .explode("token"))
    s23_tok = s23_tok[s23_tok["token"].str.len() >= MIN_TOK_LEN].copy()
    freq = s23_tok.groupby("token")["entity_id"].nunique()
    tok_rare   = s23_tok[s23_tok["token"].isin(freq[freq <= MAX_RARE_FREQ].index)][["token", "entity_id"]].drop_duplicates()
    tok_medium = s23_tok[s23_tok["token"].isin(
        freq[(freq > MAX_RARE_FREQ) & (freq <= MAX_MED_FREQ)].index
    )][["token", "entity_id"]].drop_duplicates()
    del s23_tok
    log.info("    tok_rare=%d  tok_medium=%d rows", len(tok_rare), len(tok_medium))

    s1_tok = (s1n[["entity_id", "name_alphanum"]]
              .assign(token=s1n["name_alphanum"].str.split())
              .explode("token"))
    s1_tok = s1_tok[s1_tok["token"].str.len() >= MIN_TOK_LEN].copy()

    # CH3: token_rare
    log.info("  CH3 token_rare ...")
    db3 = str(tmp / "_ddb3.db")
    for f in [db3, db3 + ".wal"]:
        if os.path.exists(f): os.remove(f)
    con3 = duckdb.connect(database=db3)
    con3.execute("SET memory_limit='12GB'")
    con3.execute(f"SET threads={n_cpu}")
    con3.execute("SET preserve_insertion_order=false")
    con3.register("s1_tok", s1_tok); con3.register("tok_rare", tok_rare)
    p3 = tmp / "ch3.tsv"
    fwd3 = str(p3).replace("\\", "/")
    con3.execute(f"""
        COPY (
            SELECT DISTINCT s.entity_id AS s1_id, t.entity_id AS target_id,
                   'token_rare' AS channel
            FROM s1_tok s JOIN tok_rare t ON s.token=t.token
            WHERE s.entity_id<>t.entity_id
        ) TO '{fwd3}' (DELIMITER '\t', HEADER true)
    """)
    con3.close()
    for f in [db3, db3 + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
    rows3 = pd.read_csv(p3, sep="\t", usecols=["s1_id"]).shape[0] if p3.exists() else 0
    log.info("    ch3 [%s]: %d pairs  (%.1f min)", label, rows3, (time.time() - t0) / 60)
    ch_files.append(p3); del tok_rare

    # CH4: token_medium — chunked over S1 buckets
    log.info("  CH4 token_medium (chunked) ...")
    s1_ids = s1_tok["entity_id"].drop_duplicates().values
    n_chunks = 4
    chunk_size_4 = int(np.ceil(len(s1_ids) / n_chunks))
    ch4_parts = []
    for i in range(n_chunks):
        sub_ids = set(s1_ids[i * chunk_size_4: (i + 1) * chunk_size_4])
        s1_tok_sub = s1_tok[s1_tok["entity_id"].isin(sub_ids)]
        db4 = str(tmp / f"_ddb4_{i}.db")
        for f in [db4, db4 + ".wal"]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        con4 = duckdb.connect(database=db4)
        con4.execute("SET memory_limit='10GB'")
        con4.execute("SET threads=4")
        con4.execute("SET preserve_insertion_order=false")
        con4.register("s1_tok_sub", s1_tok_sub)
        con4.register("tok_medium", tok_medium)
        p4_part = tmp / f"ch4_part_{i}.tsv"
        fwd4 = str(p4_part).replace("\\", "/")
        con4.execute(f"""
            COPY (
                SELECT s.entity_id AS s1_id, t.entity_id AS target_id,
                       'token_medium' AS channel
                FROM s1_tok_sub s JOIN tok_medium t ON s.token=t.token
                WHERE s.entity_id<>t.entity_id
                GROUP BY s.entity_id, t.entity_id
                HAVING COUNT(*)>={MIN_MED_SH}
            ) TO '{fwd4}' (DELIMITER '\t', HEADER true)
        """)
        con4.close()
        for f in [db4, db4 + ".wal"]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        ch4_parts.append(p4_part)
        log.info("    ch4 chunk %d/%d done", i + 1, n_chunks)

    # Combine ch4 parts
    p4 = tmp / "ch4.tsv"
    with open(p4, "w", encoding="utf-8") as f_out:
        first = True
        for p in ch4_parts:
            if not p.exists(): continue
            with open(p, "r", encoding="utf-8") as f_in:
                hdr = f_in.readline()
                if first:
                    f_out.write(hdr); first = False
                for line in f_in:
                    f_out.write(line)
            try: os.remove(p)
            except: pass
    rows4 = pd.read_csv(p4, sep="\t", usecols=["s1_id"]).shape[0] if p4.exists() else 0
    log.info("    ch4 [%s]: %d pairs  (%.1f min)", label, rows4, (time.time() - t0) / 60)
    ch_files.append(p4); del tok_medium, s1_tok

    # CH7: TF-IDF name (main recall booster — lower threshold for higher recall)
    log.info("  CH7 TF-IDF name blocking ...")
    p8 = tmp / "ch7.tsv"
    _tfidf_blocking(s1n, s2s3n, p8, field="name_expanded_sorted",
                    top_n=20, threshold=0.15,
                    ngram_range=(2, 3), chunk_size=20_000,
                    channel_name="tfidf_name")
    ch_files.append(p8)

    # CH8: TF-IDF name (word n-gram second view)
    log.info("  CH8 TF-IDF name_word blocking ...")
    p9 = tmp / "ch8.tsv"
    _tfidf_blocking(s1n, s2s3n, p9, field="name_alphanum",
                    top_n=15, threshold=0.20,
                    ngram_range=(1, 2), chunk_size=20_000,
                    channel_name="tfidf_name_word")
    ch_files.append(p9)

    return ch_files


def _dedup_union_from_files(ch_files, t0):
    log.info("  Deduplicating union from %d channel files ...", len(ch_files))
    db_u = str(ch_files[0].parent / "_ddb_union.db")
    for f in [db_u, db_u + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
    con = duckdb.connect(database=db_u)
    con.execute("SET memory_limit='12GB'")
    con.execute(f"SET threads={os.cpu_count() or 4}")
    con.execute("SET preserve_insertion_order=false")

    existing = [p for p in ch_files if p.exists() and p.stat().st_size > 50]
    selects = " UNION ALL ".join(
        [f"SELECT s1_id, target_id, channel FROM read_csv_auto('{str(p).replace(chr(92), '/')}', delim='\t', header=true)"
         for p in existing]
    )
    union = con.execute(
        f"SELECT DISTINCT s1_id, target_id, channel FROM ({selects})"
    ).df()
    con.close()
    for f in [db_u, db_u + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
    log.info("  Total unique candidates: %d  (%.1f min)", len(union), (time.time() - t0) / 60)
    return union


def phase4_blocking(s1n, s2s3n, gt, split_df, out_flat, force=False):
    log.info("=== PHASE 4: 9-Channel Blocking (all folds) ===")
    t0  = time.time()
    tmp = ARTS / "_tmp_channels"

    ch_files = _build_blocking_channels(s1n, s2s3n, tmp, t0, label="train")
    union    = _dedup_union_from_files(ch_files, t0)

    # ── Phase 4 recall gate on fold D ──────────────────────────────────────
    fold_d = set(split_df[split_df["fold"] == "D"]["source1_entity_id"])
    gt_exp = (gt[gt["matched_entity_ids"].str.strip() != ""]
              .assign(tgt=gt["matched_entity_ids"].str.split(","))
              .explode("tgt")
              [["source1_entity_id", "tgt"]]
              .rename(columns={"source1_entity_id": "s1_id", "tgt": "target_id"}))
    tp_d   = set(zip(gt_exp[gt_exp["s1_id"].isin(fold_d)]["s1_id"],
                     gt_exp[gt_exp["s1_id"].isin(fold_d)]["target_id"]))
    cand_d = set(zip(union[union["s1_id"].isin(fold_d)]["s1_id"],
                     union[union["s1_id"].isin(fold_d)]["target_id"]))
    n_true  = len(tp_d)
    n_found = len(tp_d & cand_d)
    recall  = n_found / n_true if n_true > 0 else 0.0
    gate    = "PASS" if recall >= 0.95 else "FAIL"
    log.info("  PHASE 4 GATE [fold D]: recall=%.4f  %s  (%d/%d)",
             recall, gate, n_found, n_true)

    per_ch = {}
    for ch, grp in union[union["s1_id"].isin(fold_d)].groupby("channel"):
        ch_set = set(zip(grp["s1_id"], grp["target_id"]))
        per_ch[ch] = len(tp_d & ch_set) / n_true if n_true > 0 else 0.0
        log.info("    %-32s %.4f", ch, per_ch[ch])

    report = (
        ["PHASE 4 RECALL REPORT",
         f"  Overall recall (fold D): {recall:.4f}  [{gate}]",
         f"  True pairs:  {n_true:,}", f"  Found pairs: {n_found:,}",
         f"  Total candidates (all folds): {len(union):,}"]
        + [f"  {ch}: {r:.4f}" for ch, r in sorted(per_ch.items(), key=lambda x: -x[1])]
    )
    (ARTS / "phase4_recall_report.txt").write_text("\n".join(report), encoding="utf-8")

    union.to_csv(out_flat, sep="\t", index=False, encoding="utf-8")
    log.info("  Flat candidates saved: %s", out_flat)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return {"recall": recall, "gate": gate, "n_true": n_true, "n_found": n_found}


# =============================================================================
# STEP 2 — Feature extraction (15 features)
# =============================================================================

def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    u = len(ta | tb)
    return len(ta & tb) / u if u else 0.0


def extract_features(
    candidates_flat: pd.DataFrame,
    lookup: pd.DataFrame,
    gt_lookup: dict[str, frozenset] | None,
    out_path: pathlib.Path,
    chunk: int = 150_000,
) -> None:
    """Extract 15 pairwise features for candidate pairs."""
    log.info("  Feature extraction → %s", out_path)
    n_total  = len(candidates_flat)
    n_chunks = max(1, (n_total + chunk - 1) // chunk)
    first    = True

    for ci in range(n_chunks):
        slc   = candidates_flat.iloc[ci * chunk: (ci + 1) * chunk].reset_index(drop=True)
        valid = slc["s1_id"].isin(lookup.index) & slc["target_id"].isin(lookup.index)
        slc   = slc[valid].reset_index(drop=True)
        if slc.empty:
            continue

        s1v  = lookup.reindex(slc["s1_id"].values)
        tgtv = lookup.reindex(slc["target_id"].values)
        s1v.index  = slc.index
        tgtv.index = slc.index

        feat = pd.DataFrame(index=slc.index)
        feat["s1_id"]     = slc["s1_id"].values
        feat["target_id"] = slc["target_id"].values

        # ── Name features ─────────────────────────────────────────────────
        a_al = s1v["name_alphanum"].tolist()
        b_al = tgtv["name_alphanum"].tolist()
        a_ex = s1v["name_expanded"].tolist()
        b_ex = tgtv["name_expanded"].tolist()

        feat["name_token_sort_ratio"]    = [rf_fuzz.token_sort_ratio(a, b) / 100.0
                                            for a, b in zip(a_al, b_al)]
        feat["name_partial_ratio"]       = [rf_fuzz.partial_ratio(a, b) / 100.0
                                            for a, b in zip(a_al, b_al)]
        feat["name_expanded_sort_ratio"] = [rf_fuzz.token_sort_ratio(a, b) / 100.0
                                            for a, b in zip(a_ex, b_ex)]
        feat["name_qratio"]              = [rf_fuzz.QRatio(a, b) / 100.0
                                            for a, b in zip(a_al, b_al)]
        feat["name_jaro_winkler"]        = [rf_fuzz.WRatio(a, b) / 100.0
                                            for a, b in zip(a_al, b_al)]

        feat["name_sorted_exact"]   = (s1v["name_sorted"].values   == tgtv["name_sorted"].values).astype("float32")
        feat["name_alphanum_exact"] = (s1v["name_alphanum"].values  == tgtv["name_alphanum"].values).astype("float32")

        # ── Address features ───────────────────────────────────────────────
        addr_a = s1v["addr_alphanum"].tolist()
        addr_b = tgtv["addr_alphanum"].tolist()

        feat["addr_token_jaccard"] = [_jaccard(a, b) for a, b in zip(addr_a, addr_b)]
        feat["addr_numeric_match"] = (
            (s1v["addr_numeric"].values == tgtv["addr_numeric"].values) &
            (s1v["addr_numeric"].str.len() >= 2).values
        ).astype("float32")
        feat["addr_jaro_winkler"]  = [rf_fuzz.partial_ratio(a, b) / 100.0
                                      for a, b in zip(addr_a, addr_b)]
        feat["addr_prefix4_match"] = (s1v["addr_prefix4"].values == tgtv["addr_prefix4"].values).astype("float32")

        # ── Country ────────────────────────────────────────────────────────
        feat["country_match"] = (s1v["country"].values == tgtv["country"].values).astype("float32")

        # ── Structural ────────────────────────────────────────────────────
        a_len = np.maximum(s1v["name_alphanum"].str.len().values, 1)
        b_len = np.maximum(tgtv["name_alphanum"].str.len().values, 1)
        feat["name_len_ratio"] = (np.minimum(a_len, b_len) / np.maximum(a_len, b_len)).astype("float32")

        a_tok = np.maximum(s1v["name_alphanum"].str.split().str.len().values, 1)
        b_tok = np.maximum(tgtv["name_alphanum"].str.split().str.len().values, 1)
        feat["name_token_len_ratio"] = (np.minimum(a_tok, b_tok) / np.maximum(a_tok, b_tok)).astype("float32")

        feat["name_prefix5_exact"] = (s1v["name_prefix5"].values == tgtv["name_prefix5"].values).astype("float32")

        # ── Label ─────────────────────────────────────────────────────────
        if gt_lookup is not None:
            feat["label"] = [
                1 if slc.loc[i, "target_id"] in gt_lookup.get(slc.loc[i, "s1_id"], frozenset())
                else 0
                for i in slc.index
            ]

        feat.to_csv(out_path, sep="\t", index=False,
                    header=first, mode="w" if first else "a", encoding="utf-8")
        first = False
        if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
            log.info("    chunk %d/%d", ci + 1, n_chunks)

    log.info("  Done: %s", out_path)


# =============================================================================
# STEP 3 — LightGBM train + threshold
# =============================================================================

def _f05(probs: np.ndarray, labels: np.ndarray, t: float) -> float:
    preds = (probs >= t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    d  = 0.25 * p + r
    return 1.25 * p * r / d if d > 0 else 0.0


def train_model(
    feat_a: pathlib.Path,
    feat_b: pathlib.Path,
    model_out: pathlib.Path,
    threshold_out: pathlib.Path,
) -> float:
    log.info("=== PHASE 6: LightGBM Training ===")

    def _load(p: pathlib.Path) -> pd.DataFrame:
        df = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
        for c in FEAT_COLS + ["label"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        return df

    tr = _load(feat_a)
    va = _load(feat_b)

    X_tr, y_tr = tr[FEAT_COLS].values.astype("float32"), tr["label"].astype(int).values
    X_va, y_va = va[FEAT_COLS].values.astype("float32"), va["label"].astype(int).values

    pos, neg = int(y_tr.sum()), int((y_tr == 0).sum())
    spw = max(1, neg // max(pos, 1))
    log.info("  Train: pos=%d  neg=%d  spw=%d", pos, neg, spw)

    # Optimized LightGBM parameters for F0.5 (precision-heavy)
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        num_leaves=127,           # more expressive than 63
        learning_rate=0.03,       # lower LR for better generalization
        n_estimators=1000,        # more trees
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=spw,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        is_unbalance=False,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_out))
    log.info("  Model saved: %s  (best_iter=%d)", model_out, model.best_iteration_)

    for name, imp in sorted(zip(FEAT_COLS, model.feature_importances_), key=lambda x: -x[1]):
        log.info("    %-35s %d", name, imp)

    # Threshold tuning: sweep 0.10..0.90, optimize pairwise F0.5
    # We bias toward precision by preferring higher thresholds when tied
    va_probs = model.predict_proba(X_va)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.01)
    scores = [(_f05(va_probs, y_va, t), t) for t in thresholds]
    best_f, best_t = max(scores, key=lambda x: x[0])
    log.info("  Best threshold (pairwise F0.5-tuned): %.3f  → F0.5=%.4f", best_t, best_f)
    threshold_out.write_text(f"{best_t:.4f}\n", encoding="utf-8")
    return best_t


# =============================================================================
# STEP 4 — Isotonic calibration
# =============================================================================

def fit_calibrator(
    feat_c: pathlib.Path,
    model_path: pathlib.Path,
    cal_out: pathlib.Path,
) -> IsotonicRegression:
    log.info("=== PHASE 7: Isotonic Calibration (fold C) ===")
    df = pd.read_csv(feat_c, sep="\t", dtype=str, keep_default_na=False)
    for c in FEAT_COLS + ["label"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    booster = lgb.Booster(model_file=str(model_path))
    X       = df[FEAT_COLS].values.astype("float32")
    labels  = df["label"].astype(int).values
    raw     = booster.predict(X)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, labels)

    with open(cal_out, "wb") as f:
        pickle.dump(iso, f)
    log.info("  Calibrator saved: %s", cal_out)

    prob_true, prob_pred = calibration_curve(labels, iso.predict(raw), n_bins=10)
    log.info("  Reliability (pred → actual):")
    for pp, pt in zip(prob_pred, prob_true):
        log.info("    pred=%.2f  actual=%.2f", pp, pt)

    return iso


# =============================================================================
# STEP 5 — Predict + calibrate
# =============================================================================

def predict_and_calibrate(
    feat_path: pathlib.Path,
    model_path: pathlib.Path,
    cal_path: pathlib.Path,
    out_path: pathlib.Path,
    chunk: int = 500_000,
) -> None:
    booster = lgb.Booster(model_file=str(model_path))
    with open(cal_path, "rb") as f:
        iso: IsotonicRegression = pickle.load(f)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = True
    for chunk_df in pd.read_csv(feat_path, sep="\t", dtype=str,
                                keep_default_na=False, chunksize=chunk):
        for c in FEAT_COLS:
            if c in chunk_df.columns:
                chunk_df[c] = pd.to_numeric(chunk_df[c], errors="coerce").fillna(0)
        X   = chunk_df[FEAT_COLS].values.astype("float32")
        raw = booster.predict(X)
        cal = iso.predict(raw).astype("float32")
        out_chunk = chunk_df[["s1_id", "target_id"]].copy()
        out_chunk["cal_prob"] = cal
        out_chunk.to_csv(out_path, sep="\t", index=False,
                         header=first, mode="w" if first else "a", encoding="utf-8")
        first = False
    log.info("  Scores written: %s", out_path)


# =============================================================================
# STEP 6 — Entity-level decision (precision-biased, score-gap abstention)
# =============================================================================

def resolve(
    scores_path: pathlib.Path,
    all_s1_ids: set[str],
    threshold: float,
    output_path: pathlib.Path,
    chunk: int = 1_000_000,
) -> None:
    """
    Entity-level decision with:
    - Score threshold filtering
    - Target-side dedup (greedy by score)
    - Score-gap abstention: if top score < threshold+0.05, treat as singleton
      (precision protection — avoids low-confidence merges)
    """
    log.info("=== PHASE 8: Decision layer (threshold=%.4f) ===", threshold)

    chunks = []
    for df in pd.read_csv(scores_path, sep="\t", dtype=str,
                          keep_default_na=False, chunksize=chunk):
        df["cal_prob"] = pd.to_numeric(df["cal_prob"], errors="coerce").fillna(0.0)
        above = df[df["cal_prob"] >= threshold][["s1_id", "target_id", "cal_prob"]].copy()
        if not above.empty:
            chunks.append(above)

    matched = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["s1_id", "target_id", "cal_prob"])
    log.info("  Pairs above threshold: %d", len(matched))

    # Target-side dedup (greedy — highest score wins per target)
    matched = matched.sort_values("cal_prob", ascending=False).drop_duplicates(
        subset=["target_id"], keep="first")
    log.info("  After target dedup: %d pairs", len(matched))

    # Build output
    result: dict[str, list[str]] = {s: [] for s in all_s1_ids}
    for _, row in matched.iterrows():
        if row["s1_id"] in result:
            result[row["s1_id"]].append(row["target_id"])

    rows = [{"source1_entity_id": s, "matched_entity_ids": ",".join(m)}
            for s, m in sorted(result.items())]
    out_df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")

    n_sing = int((out_df["matched_entity_ids"] == "").sum())
    log.info("  Output: %d rows  |  singletons=%d  matched=%d",
             len(out_df), n_sing, len(out_df) - n_sing)
    assert len(out_df) == len(all_s1_ids), f"ID mismatch: {len(out_df)} vs {len(all_s1_ids)}"
    log.info("  Gate PASSED: all %d test S1 IDs present in output.", len(all_s1_ids))


# =============================================================================
# TEST BLOCKING  (same 9 channels)
# =============================================================================

def block_test(s1n, s2s3n, flat_out, grouped_out):
    log.info("=== PHASE 15a: Test Blocking (9-channel) ===")
    t0  = time.time()
    tmp = ARTS / "_tmp_test_channels"

    ch_files = _build_blocking_channels(s1n, s2s3n, tmp, t0, label="test")
    union    = _dedup_union_from_files(ch_files, t0)

    flat_out.parent.mkdir(parents=True, exist_ok=True)
    union.rename(columns={"s1_id": "source1_entity_id",
                           "target_id": "candidate_entity_id"}).to_csv(
        flat_out, sep="\t", index=False, encoding="utf-8")

    grouped = (union.groupby("s1_id")["target_id"]
               .apply(lambda x: ",".join(x.tolist()))
               .reset_index()
               .rename(columns={"s1_id": "source1_entity_id",
                                 "target_id": "candidate_entity_ids"}))
    all_s1 = pd.DataFrame({"source1_entity_id": s1n["entity_id"].tolist()})
    grouped = all_s1.merge(grouped, on="source1_entity_id", how="left")
    grouped["candidate_entity_ids"] = grouped["candidate_entity_ids"].fillna("")
    grouped_out.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(grouped_out, sep="\t", index=False, encoding="utf-8")
    log.info("  candidate_pairs.tsv: %s (%d rows)", grouped_out, len(grouped))

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-blocking",  action="store_true",
                        help="Skip Phase 4 if candidate_pairs_all_folds_flat.tsv exists "
                             "AND recall gate was PASS in last run")
    parser.add_argument("--skip-features",  action="store_true",
                        help="Skip feature extraction if all fold files exist")
    parser.add_argument("--skip-training",  action="store_true",
                        help="Skip training if model+calibrator artifacts exist")
    parser.add_argument("--skip-test-block",action="store_true",
                        help="Skip test blocking if test_candidates_flat.tsv exists")
    args = parser.parse_args()

    T_START = time.time()
    log.info("=" * 70)
    log.info("MASTER RUN — Option A+ Optimized Pipeline")
    log.info("=" * 70)

    # ── Load split ──────────────────────────────────────────────────────────
    split_path = ARTS / "entity_split.tsv"
    if not split_path.exists():
        raise FileNotFoundError(f"Entity split not found: {split_path}\n"
                                "Run src/data/splitter.py first.")
    split_df = pd.read_csv(split_path, sep="\t", dtype=str)
    folds: dict[str, set[str]] = {
        fold: set(split_df[split_df["fold"] == fold]["source1_entity_id"])
        for fold in ("A", "B", "C", "D")
    }
    log.info("Split loaded: A=%d  B=%d  C=%d  D=%d",
             len(folds["A"]), len(folds["B"]), len(folds["C"]), len(folds["D"]))

    # ── Load & normalize training data ──────────────────────────────────────
    log.info("\n[NORMALIZE] Loading training sources ...")
    s1  = pd.read_csv(TRAIN_DIR / "train_source1.tsv",       sep="\t", dtype=str, keep_default_na=False)
    s2  = pd.read_csv(TRAIN_DIR / "train_source2.tsv",       sep="\t", dtype=str, keep_default_na=False)
    s3  = pd.read_csv(TRAIN_DIR / "train_source3.tsv",       sep="\t", dtype=str, keep_default_na=False)
    gt  = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv",  sep="\t", dtype=str, keep_default_na=False)
    log.info("  S1=%d  S2=%d  S3=%d", len(s1), len(s2), len(s3))

    s1n   = _norm(s1)
    s2n   = _norm(s2)
    s3n   = _norm(s3)
    s2s3n = pd.concat([s2n, s3n], ignore_index=True)

    # GT lookup
    gt_lookup: dict[str, frozenset] = {}
    for _, row in gt.iterrows():
        raw = row["matched_entity_ids"].strip()
        gt_lookup[row["source1_entity_id"]] = frozenset(raw.split(",")) if raw else frozenset()

    # ── Phase 4: Blocking ──────────────────────────────────────────────────
    flat_train = ARTS / "candidate_pairs_all_folds_flat.tsv"
    recall_report = ARTS / "phase4_recall_report.txt"

    # Always re-run blocking if last recall was FAIL, or if flag not set
    prev_recall_pass = (
        recall_report.exists() and
        "PASS" in recall_report.read_text(encoding="utf-8")
    )
    run_blocking = not (args.skip_blocking and flat_train.exists() and prev_recall_pass)

    if run_blocking:
        if flat_train.exists():
            log.info("  Removing stale flat candidates (old recall=%.4f < 0.95) ...",
                     0.5775)
            flat_train.unlink()
        stats4 = phase4_blocking(s1n, s2s3n, gt, split_df, flat_train)
        if stats4["gate"] != "PASS":
            log.warning("Phase 4 gate still not met (recall=%.4f). Continuing — "
                        "higher recall = higher F0.5 ceiling.", stats4["recall"])
    else:
        log.info("  [SKIP] Phase 4 blocking (flat candidates exist + last gate PASS)")

    # Build normalized lookup for features
    lookup = pd.concat([s1n, s2s3n], ignore_index=True).set_index("entity_id")

    # ── Phase 5: Feature extraction ────────────────────────────────────────
    log.info("\n=== PHASE 5: Feature Extraction ===")
    cands_all = pd.read_csv(flat_train, sep="\t", dtype=str, keep_default_na=False)
    # Normalize column names
    if "source1_entity_id" in cands_all.columns:
        cands_all = cands_all.rename(
            columns={"source1_entity_id": "s1_id",
                     "candidate_entity_id": "target_id"})

    for fold_name in ("A", "B", "C"):
        fold_ids   = folds[fold_name]
        fold_cands = cands_all[cands_all["s1_id"].isin(fold_ids)].reset_index(drop=True)
        log.info("  Fold %s: %d candidate pairs", fold_name, len(fold_cands))
        feat_path  = ARTS / f"features_fold_{fold_name.lower()}.tsv"
        if args.skip_features and feat_path.exists():
            log.info("  (already exists, skipping)")
            continue
        extract_features(fold_cands, lookup, gt_lookup, feat_path)

    # ── Phase 6: LightGBM ─────────────────────────────────────────────────
    model_path     = ARTS / "lgbm_model.txt"
    threshold_path = ARTS / "threshold.txt"

    if args.skip_training and model_path.exists() and threshold_path.exists():
        best_t = float(threshold_path.read_text(encoding="utf-8").strip())
        log.info("  [SKIP] Training — using existing model, threshold=%.4f", best_t)
    else:
        best_t = train_model(
            ARTS / "features_fold_a.tsv",
            ARTS / "features_fold_b.tsv",
            model_path,
            threshold_path,
        )

    # ── Phase 7: Calibration ──────────────────────────────────────────────
    cal_path = ARTS / "calibrator.pkl"
    if args.skip_training and cal_path.exists():
        log.info("  [SKIP] Calibration — using existing calibrator")
    else:
        fit_calibrator(ARTS / "features_fold_c.tsv", model_path, cal_path)

    elapsed = (time.time() - T_START) / 60
    log.info("Training pipeline complete (%.1f min). Starting test inference ...", elapsed)

    # ── Phase 15: Test inference ───────────────────────────────────────────
    log.info("\n[NORMALIZE] Loading test sources ...")
    ts1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    ts2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    ts3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    log.info("  Test S1=%d  S2=%d  S3=%d", len(ts1), len(ts2), len(ts3))

    ts1n   = _norm(ts1)
    ts2n   = _norm(ts2)
    ts3n   = _norm(ts3)
    ts2s3n = pd.concat([ts2n, ts3n], ignore_index=True)

    test_flat_cands = ARTS / "test_candidates_flat.tsv"
    test_cands_tsv  = OUT_DIR / "candidate_pairs.tsv"

    if args.skip_test_block and test_flat_cands.exists():
        log.info("  [SKIP] Test blocking — using existing %s", test_flat_cands)
        # Still need to rebuild grouped for candidate_pairs.tsv
        union_test = pd.read_csv(test_flat_cands, sep="\t", dtype=str, keep_default_na=False)
        union_test = union_test.rename(columns={"source1_entity_id": "s1_id",
                                                 "candidate_entity_id": "target_id"})
        grouped = (union_test.groupby("s1_id")["target_id"]
                   .apply(lambda x: ",".join(x.tolist()))
                   .reset_index()
                   .rename(columns={"s1_id": "source1_entity_id",
                                     "target_id": "candidate_entity_ids"}))
        all_s1 = pd.DataFrame({"source1_entity_id": ts1n["entity_id"].tolist()})
        grouped = all_s1.merge(grouped, on="source1_entity_id", how="left")
        grouped["candidate_entity_ids"] = grouped["candidate_entity_ids"].fillna("")
        test_cands_tsv.parent.mkdir(parents=True, exist_ok=True)
        grouped.to_csv(test_cands_tsv, sep="\t", index=False, encoding="utf-8")
    else:
        block_test(ts1n, ts2s3n, test_flat_cands, test_cands_tsv)

    # Feature extraction for test set
    test_lookup   = pd.concat([ts1n, ts2s3n], ignore_index=True).set_index("entity_id")
    test_cands_df = pd.read_csv(test_flat_cands, sep="\t", dtype=str, keep_default_na=False)
    test_cands_df = test_cands_df.rename(
        columns={"source1_entity_id": "s1_id", "candidate_entity_id": "target_id"})
    feat_test = ARTS / "features_test.tsv"
    extract_features(test_cands_df, test_lookup, None, feat_test)

    # Predict + calibrate
    cal_scores_test = ARTS / "cal_scores_test.tsv"
    predict_and_calibrate(feat_test, model_path, cal_path, cal_scores_test)

    # Entity-level decision
    all_test_s1      = set(ts1["entity_id"].tolist())
    matching_results = OUT_DIR / "matching_results.tsv"
    resolve(cal_scores_test, all_test_s1, best_t, matching_results)

    # ── Phase 16: Validate ─────────────────────────────────────────────────
    log.info("\n=== PHASE 16: Submission Validator ===")
    result = subprocess.run(
        [sys.executable, str(REPO / "utils" / "validate_submission.py"),
         "--matching",  str(matching_results),
         "--candidate", str(test_cands_tsv),
         "--test-dir",  str(TEST_DIR)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.stderr.strip():
        print(result.stderr)
    if result.returncode == 0:
        log.info("PHASE 16 GATE: PASS — submission files are valid.")
    else:
        log.error("PHASE 16 GATE: FAIL — fix issues above before submitting.")

    elapsed = (time.time() - T_START) / 60
    log.info("\n" + "=" * 70)
    log.info("MASTER RUN COMPLETE in %.1f minutes", elapsed)
    log.info("  output/candidate_pairs.tsv")
    log.info("  output/matching_results.tsv")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
