#!/usr/bin/env python
"""
add_new_channels.py — Add CH6-CH8 to existing CH1-CH5, re-dedup, save flat candidates.

REVISED: removed CH6 (name_prefix5) which caused cardinality explosion.
New channels:
  CH6: addr_numeric + country (4+ digit door/building/PIN number + country)
  CH7: TF-IDF cosine on name_expanded_sorted (char n-gram 2-3), top-20, thresh=0.15
       (lower threshold and more top-N to maximize recall)
  CH8: TF-IDF cosine on name_alphanum (char n-gram 1-2), top-10, thresh=0.20
       (second name view for extra coverage)

Run from repo root:
    python add_new_channels.py
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))

def _ensure(pkg, name=None):
    import importlib.util
    if importlib.util.find_spec(name or pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

_ensure("pyyaml",         "yaml")
_ensure("rapidfuzz",      "rapidfuzz")
_ensure("duckdb",         "duckdb")
_ensure("sparse-dot-topn","sparse_dot_topn")

import numpy as np
import pandas as pd
import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("add_channels")

from config.config import load_config

CFG       = load_config()
ARTS      = REPO / CFG["paths"]["artifacts_dir"]
TRAIN_DIR = REPO / CFG["paths"]["train_dir"]

TMP = ARTS / "_tmp_channels"
TMP.mkdir(parents=True, exist_ok=True)


# ── Normalization ────────────────────────────────────────────────────────────

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
}


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    name = df["business_name"].fillna("").str.lower().str.strip()
    name = name.str.replace(r"<null>", "", regex=True)
    name = name.str.replace(r"^(smt|sri|--|[*]+)\s+", "", regex=True)
    name = name.str.replace(r"[^\w\s&]", " ", regex=True)
    name = name.str.replace(r"\s+", " ", regex=True).str.strip()

    alnum = name.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    alnum = alnum.str.replace(r"\s+", " ", regex=True).str.strip()

    sorted_view = alnum.str.split().apply(
        lambda t: " ".join(sorted(t)) if t else "")

    exp = alnum.copy()
    for src, tgt in _ABBR_MAP.items():
        exp = exp.str.replace(rf"\b{src}\b", tgt, regex=True)
    exp = exp.str.replace(r"\s+", " ", regex=True).str.strip()
    exp_sorted = exp.str.split().apply(lambda t: " ".join(sorted(t)) if t else "")

    addr = df["business_address"].fillna("").str.lower().str.strip()
    addr_alnum = addr.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    addr_alnum = addr_alnum.str.replace(r"\s+", " ", regex=True).str.strip()
    addr_num   = addr_alnum.str.extract(r"(\d+)", expand=False).fillna("")

    out = df[["entity_id", "business_name", "business_address", "country"]].copy()
    out["name_alphanum"]        = alnum
    out["name_sorted"]          = sorted_view
    out["name_expanded"]        = exp
    out["name_expanded_sorted"] = exp_sorted
    out["addr_alphanum"]        = addr_alnum
    out["addr_numeric"]         = addr_num
    return out


# ── TF-IDF blocking helper ──────────────────────────────────────────────────

def _tfidf_blocking(
    s1n: pd.DataFrame, s2s3n: pd.DataFrame,
    out_path: pathlib.Path, field: str,
    top_n: int, threshold: float,
    ngram_range=(2, 3), chunk_size: int = 20_000,
    channel_name: str = "tfidf",
) -> int:
    log.info("  TF-IDF [%s]: field=%s top_n=%d thresh=%.2f ngram=%s",
             channel_name, field, top_n, threshold, ngram_range)

    s1_texts   = s1n[field].fillna("").tolist()
    s2s3_texts = s2s3n[field].fillna("").tolist()
    s1_ids     = s1n["entity_id"].tolist()
    s2s3_ids   = s2s3n["entity_id"].tolist()

    corpus = s1_texts + s2s3_texts
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngram_range,
        min_df=2, max_df=0.95, sublinear_tf=True,
    )
    vec.fit(corpus)
    log.info("    Vocabulary size: %d", len(vec.vocabulary_))

    s2s3_mat = vec.transform(s2s3_texts)
    s1_mat   = vec.transform(s1_texts)
    log.info("    S1 matrix: %s  S2S3 matrix: %s", s1_mat.shape, s2s3_mat.shape)

    n_s1  = len(s1_ids)
    first = True
    total = 0
    n_cpu = max(1, (os.cpu_count() or 4) // 2)

    for i, start in enumerate(range(0, n_s1, chunk_size)):
        end       = min(start + chunk_size, n_s1)
        chunk_mat = s1_mat[start:end]
        matches   = awesome_cossim_topn(
            chunk_mat, s2s3_mat.T,
            ntop=top_n, lower_bound=threshold,
            use_threads=True, n_threads=n_cpu,
        )
        cx = matches.tocoo()
        if cx.nnz == 0:
            continue

        pairs = pd.DataFrame({
            "s1_id":     [s1_ids[start + r] for r in cx.row],
            "target_id": [s2s3_ids[c]        for c in cx.col],
            "channel":   channel_name,
        })
        pairs = pairs[pairs["s1_id"] != pairs["target_id"]]
        pairs.to_csv(
            out_path, sep="\t", index=False,
            header=first, mode="w" if first else "a", encoding="utf-8",
        )
        first = False
        total += len(pairs)

        if (i + 1) % 20 == 0:
            log.info("    chunk %d/%d  total=%d  (%.1f min)",
                     i + 1, -(-n_s1 // chunk_size), total,
                     (time.time() - _T0) / 60)

    if first:
        pd.DataFrame(columns=["s1_id", "target_id", "channel"]).to_csv(
            out_path, sep="\t", index=False, encoding="utf-8"
        )

    log.info("    [%s]: %d pairs  (%.1f min)", channel_name, total,
             (time.time() - _T0) / 60)
    return total


# ── DuckDB addr_numeric channel ─────────────────────────────────────────────

def _addr_numeric_channel(s1n: pd.DataFrame, s2s3n: pd.DataFrame) -> pathlib.Path:
    """CH6: exact match on addr_numeric (≥4 digits) + country."""
    p6 = TMP / "ch6.tsv"
    if p6.exists():
        log.info("  CH6 already exists, skipping")
        return p6

    log.info("  CH6 addr_numeric_country (DuckDB) ...")
    db_path = str(TMP / "_ddb_ch6.db")
    for f in [db_path, db_path + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    con = duckdb.connect(database=db_path)
    con.execute("SET memory_limit='10GB'")
    con.execute(f"SET threads={max(4, (os.cpu_count() or 8) // 2)}")
    con.execute("SET preserve_insertion_order=false")
    con.register("s1n",   s1n)
    con.register("s2s3n", s2s3n)

    fwd6 = str(p6).replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id,
                   'addr_numeric_country' AS channel
            FROM s1n a JOIN s2s3n b ON a.addr_numeric = b.addr_numeric
                                   AND a.country = b.country
            WHERE length(a.addr_numeric) >= 4
              AND a.entity_id <> b.entity_id
        ) TO '{fwd6}' (DELIMITER '\t', HEADER true)
    """)
    con.close()
    for f in [db_path, db_path + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    try:
        n6 = pd.read_csv(p6, sep="\t", usecols=["s1_id"]).shape[0]
        log.info("    ch6: %d pairs  (%.1f min)", n6, (time.time() - _T0) / 60)
    except Exception:
        log.info("    ch6: 0 pairs")
    return p6


# ── Union dedup ──────────────────────────────────────────────────────────────

def dedup_union(ch_files: list[pathlib.Path], out_flat: pathlib.Path) -> pd.DataFrame:
    log.info("  Deduplicating union from %d channel files ...", len(ch_files))
    existing = [p for p in ch_files if p.exists() and p.stat().st_size > 50]
    log.info("  Usable files (%d): %s", len(existing),
             [p.stem for p in existing])

    db_u = str(TMP / "_ddb_union2.db")
    for f in [db_u, db_u + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    con = duckdb.connect(database=db_u)
    con.execute("SET memory_limit='14GB'")
    con.execute(f"SET threads={os.cpu_count() or 4}")
    con.execute("SET preserve_insertion_order=false")

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

    log.info("  Total unique candidates after dedup: %d  (%.1f min)",
             len(union), (time.time() - _T0) / 60)
    union.to_csv(out_flat, sep="\t", index=False, encoding="utf-8")
    log.info("  Written → %s", out_flat)
    return union


# ── Recall gate check ────────────────────────────────────────────────────────

def check_recall(union: pd.DataFrame, split_df: pd.DataFrame, gt: pd.DataFrame):
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
    log.info("  RECALL CHECK [fold D]: recall=%.4f  %s  (%d/%d)",
             recall, gate, n_found, n_true)

    per_ch = {}
    for ch, grp in union[union["s1_id"].isin(fold_d)].groupby("channel"):
        ch_set = set(zip(grp["s1_id"], grp["target_id"]))
        per_ch[ch] = len(tp_d & ch_set) / n_true if n_true > 0 else 0.0
        log.info("    %-32s %.4f", ch, per_ch[ch])

    # Per-channel additive contribution
    found_so_far = set()
    sorted_chs = sorted(per_ch.items(), key=lambda x: -x[1])
    for ch, _ in sorted_chs:
        ch_grp = union[union["s1_id"].isin(fold_d) & (union["channel"] == ch)]
        ch_set = set(zip(ch_grp["s1_id"], ch_grp["target_id"]))
        new_found = (tp_d & ch_set) - found_so_far
        found_so_far |= new_found
        log.info("    %-32s marginal=%d  cumulative=%.4f", ch, len(new_found),
                 len(found_so_far) / n_true if n_true > 0 else 0)

    report = (
        ["PHASE 4 RECALL REPORT",
         f"  Overall recall (fold D): {recall:.4f}  [{gate}]",
         f"  True pairs:  {n_true:,}", f"  Found pairs: {n_found:,}",
         f"  Total candidates (all folds): {len(union):,}"]
        + [f"  {ch}: {r:.4f}" for ch, r in sorted(per_ch.items(), key=lambda x: -x[1])]
    )
    (ARTS / "phase4_recall_report.txt").write_text("\n".join(report), encoding="utf-8")
    log.info("  Report saved → %s", ARTS / "phase4_recall_report.txt")
    return recall, gate


_T0 = time.time()


def main():
    global _T0
    _T0 = time.time()
    log.info("=" * 60)
    log.info("ADD NEW CHANNELS (CH6-CH8) — Incremental Blocking")
    log.info("=" * 60)

    # Load data
    log.info("[1/5] Loading training data ...")
    s1  = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2  = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3  = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt  = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    split_df = pd.read_csv(ARTS / "entity_split.tsv", sep="\t", dtype=str)
    log.info("  S1=%d  S2=%d  S3=%d", len(s1), len(s2), len(s3))

    s1n   = _norm(s1)
    s2n   = _norm(s2)
    s3n   = _norm(s3)
    s2s3n = pd.concat([s2n, s3n], ignore_index=True)
    log.info("  Normalized. (%.1f min)", (time.time() - _T0) / 60)

    # CH6: addr_numeric + country (DuckDB, fast, precise)
    log.info("[2/5] CH6: addr_numeric_country ...")
    _addr_numeric_channel(s1n, s2s3n)

    # CH7: TF-IDF name (main recall booster)
    log.info("[3/5] CH7: TF-IDF name (top-20, thresh=0.15) ...")
    p7 = TMP / "ch7.tsv"
    if not p7.exists():
        _tfidf_blocking(s1n, s2s3n, p7,
                        field="name_expanded_sorted",
                        top_n=20, threshold=0.15,
                        ngram_range=(2, 3), chunk_size=20_000,
                        channel_name="tfidf_name")
    else:
        log.info("  CH7 already exists, skipping")

    # CH8: TF-IDF name (second view with word n-grams for additional coverage)
    log.info("[4/5] CH8: TF-IDF name alphanum (top-15, thresh=0.20, word ngrams) ...")
    p8 = TMP / "ch8.tsv"
    if not p8.exists():
        _tfidf_blocking(s1n, s2s3n, p8,
                        field="name_alphanum",
                        top_n=15, threshold=0.20,
                        ngram_range=(1, 2), chunk_size=20_000,
                        channel_name="tfidf_name_word")
    else:
        log.info("  CH8 already exists, skipping")

    # Dedup union of all 8 channels
    log.info("[5/5] Deduplicating union of all channels ...")
    all_ch_files = [
        TMP / "ch1.tsv", TMP / "ch2.tsv", TMP / "ch3.tsv",
        TMP / "ch4.tsv", TMP / "ch5.tsv", TMP / "ch6.tsv",
        TMP / "ch7.tsv", TMP / "ch8.tsv",
    ]
    flat_out = ARTS / "candidate_pairs_all_folds_flat.tsv"
    union = dedup_union(all_ch_files, flat_out)

    # Check recall
    log.info("Checking recall gate ...")
    recall, gate = check_recall(union, split_df, gt)

    elapsed = (time.time() - _T0) / 60
    log.info("=" * 60)
    log.info("DONE in %.1f min — recall=%.4f  [%s]", elapsed, recall, gate)
    if gate == "FAIL":
        log.warning("Recall below 0.95. Consider lowering TF-IDF threshold further.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
