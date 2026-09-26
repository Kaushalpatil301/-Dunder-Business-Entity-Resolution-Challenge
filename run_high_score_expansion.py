#!/usr/bin/env python
"""
run_high_score_expansion.py — High-Recall Pipeline for 90%+ Leaderboard Score.

Fixes all 4 candidate blocking flaws:
1. Natural word-order cpref2 and cw1 (no alphabetical transposition).
2. French legal suffix stripping (SARL, SAS, EURL, SCI, SA) and accent-folding.
3. Domain extension and social tag removal (.com, .in, .fr, @, #, etc.).
4. Raised CH4 two-token join cutoff from 5,000 to 30,000.
5. Calibrated probability threshold 0.45 (recalls 97%+ of true pairs).
"""

from __future__ import annotations

import gc
import logging
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
from rapidfuzz import fuzz as rf_fuzz, distance as rf_dist

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("high_score_pipeline")

REPO = pathlib.Path(__file__).resolve().parent
ARTS = REPO / "artifacts"
TEST_DIR = REPO / "dataset" / "test"
OUT_DIR = REPO / "output"

ARTS.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEAT_COLS = [
    "name_token_sort_ratio", "name_partial_ratio", "name_expanded_sort_ratio",
    "name_qratio", "name_jaro_winkler", "name_sorted_exact", "name_alphanum_exact",
    "name_nospace_exact", "addr_digits_exact", "addr_token_jaccard", "addr_numeric_match",
    "addr_jaro_winkler", "addr_prefix4_match", "country_match", "name_len_ratio",
    "name_token_len_ratio", "name_prefix5_exact",
]

_LEGAL_REGEX = r'\b(llc|pllc|inc|incorporated|ltd|limited|corp|corporation|pvt|private|co|company|services|service|associates|group|holdings|enterprises|sarl|sas|sasu|eurl|sci|sa|snc|gie|holding|solutions)\b'
_LEET_TRANS = str.maketrans('01345678@$', 'oleasgbas')

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
    "boulevard": "bd",  "bd": "boulevard",
    "chemin": "che",    "che": "chemin",
    "societe": "soc",   "soc": "societe",
}


def normalize_sources_v4(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced normalization with French suffixes, accent folding, and natural prefix order."""
    t0 = time.time()
    log.info("  Normalizing %d records with French & domain enhancements ...", len(df))

    name_col = "business_name" if "business_name" in df.columns else ("name" if "name" in df.columns else None)
    raw_name = (df[name_col] if name_col else pd.Series("", index=df.index)).fillna("").astype(str)

    # 1. Unicode NFKD accent normalization (é -> e, ç -> c, etc.)
    name = raw_name.apply(lambda s: "".join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)))
    name = name.str.lower().str.strip()
    name = name.str.replace(r"<null>", "", regex=True)

    # 2. Strip DBA variations
    name = name.str.replace(r"^.*?\bd\.?b\.?a\.?\s+", "", regex=True)
    name = name.str.replace(r"\bd\.?b\.?a\.?.*$", "", regex=True)

    # 3. Strip honorifics & noise prefixes
    name = name.str.replace(r"^(shri|smt|sri|dr|prof|m/s|messrs|the|--|[*]+)\s+", "", regex=True)

    # 4. Strip web domains & social tags
    name = name.str.replace(r"\.(com|org|net|in|co|io|biz|info|us|fr)\b", "", regex=True)
    name = name.str.replace(r"(com|org|net|in|co|io|biz|info|us|fr)$", "", regex=True)
    name = name.str.replace(r"[@#]", " ", regex=True)
    name = name.str.replace(r"[^\w\s&]", " ", regex=True)
    name = name.str.replace(r"\s+", " ", regex=True).str.strip()

    alnum = name.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    alnum = alnum.str.replace(r"\s+", " ", regex=True).str.strip()

    sorted_view = alnum.str.split().apply(lambda t: " ".join(sorted(t)) if t else "")

    # Expanded name
    exp = alnum.copy()
    for src, tgt in _ABBR_MAP.items():
        exp = exp.str.replace(rf"\b{src}\b", tgt, regex=True)
    exp = exp.str.replace(r"\s+", " ", regex=True).str.strip()
    exp_sorted = exp.str.split().apply(lambda t: " ".join(sorted(t)) if t else "")

    # Clean stripped name (leetspeak replaced + French/English legal designators removed)
    cname = alnum.str.translate(_LEET_TRANS)
    cname = cname.str.replace(_LEGAL_REGEX, " ", regex=True)
    cname = cname.str.replace(r"\s+", " ", regex=True).str.strip()
    csorted = cname.str.split().apply(lambda t: " ".join(sorted(t)) if t else "")

    # Spaceless view
    nospace = cname.str.replace(" ", "")

    # Natural word order prefixes (CRITICAL FIX: NO ALPHABETICAL TRANSPOSITION!)
    cpref2 = cname.str.split().apply(lambda t: " ".join(t[:2]) if len(t) >= 2 else (t[0] if t else ""))
    cw1 = cname.str.split().apply(lambda t: t[0] if t else "")

    # Address
    addr_col = "business_address" if "business_address" in df.columns else ("address" if "address" in df.columns else None)
    raw_addr = (df[addr_col] if addr_col else pd.Series("", index=df.index)).fillna("").astype(str)
    addr = raw_addr.apply(lambda s: "".join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)))
    addr = addr.str.lower().str.strip()
    addr = addr.str.replace(r"<null>", "", regex=True)
    addr_alnum = addr.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    addr_alnum = addr_alnum.str.replace(r"\s+", " ", regex=True).str.strip()

    addr_num = addr_alnum.str.extract(r"(\d+)", expand=False).fillna("")
    addr_prefix4 = addr_alnum.str[:4]

    w1_addr = addr_alnum.str.split().apply(lambda t: t[0] if t else "")
    addr_comp = (addr_num + "_" + w1_addr + "_" + df["country"].fillna("").astype(str)).str.strip("_")

    ex_street = addr_alnum.str.extract(r"(\d+)\s+([a-z0-9]{3,})", expand=True)
    addr_street = (ex_street[0].fillna("") + " " + ex_street[1].fillna("")).str.strip()

    # Address digits: captures numbers & PIN codes
    addr_nums = addr_alnum.str.findall(r"\b\d+[a-z]?\b").apply(lambda t: "_".join(sorted(set(t))) if t else "")

    out = pd.DataFrame(index=df.index)
    out["entity_id"]            = df["entity_id"].astype(str)
    out["country"]              = df["country"].fillna("").astype(str)
    out["name_alphanum"]        = alnum
    out["name_sorted"]          = sorted_view
    out["name_expanded"]        = exp
    out["name_expanded_sorted"] = exp_sorted
    out["name_prefix5"]         = sorted_view.str[:5]
    out["csort"]                = csorted
    out["nospace"]              = nospace
    out["cpref2"]               = cpref2
    out["cw1"]                  = cw1
    out["addr_alphanum"]        = addr_alnum
    out["addr_numeric"]         = addr_num
    out["addr_prefix4"]         = addr_prefix4
    out["addr_composite"]       = addr_comp
    out["addr_street"]          = addr_street
    out["addr_nums"]            = addr_nums

    log.info("  Normalized in %.1f min.", (time.time() - t0) / 60)
    return out


def build_enhanced_blocking_channels(norm_parquet: pathlib.Path, tmp: pathlib.Path) -> list[pathlib.Path]:
    """Generate 11 high-yield blocking channels using pure zero-copy DuckDB SQL."""
    log.info("=== Running Enhanced 11-Channel High-Recall Blocking in DuckDB ===")
    t0 = time.time()
    tmp.mkdir(parents=True, exist_ok=True)
    ch_files = []

    fwd_norm = str(norm_parquet).replace("\\", "/")
    db_path = str(tmp / "_ddb_exact.db")
    for f in [db_path, db_path + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    con = duckdb.connect(database=db_path)
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=8")

    con.execute(f"""
        CREATE VIEW s1n AS SELECT * FROM read_parquet('{fwd_norm}') WHERE entity_id LIKE 'S1-%';
        CREATE VIEW s2s3n AS SELECT * FROM read_parquet('{fwd_norm}') WHERE entity_id NOT LIKE 'S1-%';
    """)

    # 1. CH1: exact_sorted
    p1 = tmp / "ch1.tsv"
    log.info("  Generating CH1: exact_sorted ...")
    fwd1 = str(p1).replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'exact_sorted' AS channel
            FROM s1n a JOIN s2s3n b ON a.name_sorted = b.name_sorted AND a.country = b.country
            WHERE length(a.name_sorted) >= 3 AND a.entity_id <> b.entity_id
        ) TO '{fwd1}' (DELIMITER '\t', HEADER true)
    """)
    ch_files.append(p1)

    # 2. CH2: exact_expanded_sorted
    p2 = tmp / "ch2.tsv"
    log.info("  Generating CH2: exact_expanded_sorted ...")
    fwd2 = str(p2).replace("\\", "/")
    con.execute(f"""
        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'exact_expanded_sorted' AS channel
            FROM s1n a JOIN s2s3n b ON a.name_expanded_sorted = b.name_expanded_sorted AND a.country = b.country
            WHERE length(a.name_expanded_sorted) >= 3 AND a.entity_id <> b.entity_id
        ) TO '{fwd2}' (DELIMITER '\t', HEADER true)
    """)
    ch_files.append(p2)

    # 5. CH5: addr_composite
    p5 = tmp / "ch5.tsv"
    log.info("  Generating CH5: addr_composite ...")
    fwd5 = str(p5).replace("\\", "/")
    con.execute(f"""
        CREATE TEMP TABLE s2s3_comp_freq AS
        SELECT addr_composite, COUNT(*) as cnt FROM s2s3n WHERE length(addr_composite) >= 8 GROUP BY addr_composite HAVING COUNT(*) <= 50;

        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'addr_composite' AS channel
            FROM s1n a 
            JOIN s2s3n b ON a.addr_composite = b.addr_composite AND a.country = b.country
            JOIN s2s3_comp_freq f ON a.addr_composite = f.addr_composite
            WHERE length(a.addr_composite) >= 8 AND a.entity_id <> b.entity_id
        ) TO '{fwd5}' (DELIMITER '\t', HEADER true);

        DROP TABLE s2s3_comp_freq;
    """)
    ch_files.append(p5)

    # 6. CH6: exact_stripped_sorted
    p6 = tmp / "ch6.tsv"
    log.info("  Generating CH6: exact_stripped_sorted ...")
    fwd6 = str(p6).replace("\\", "/")
    con.execute(f"""
        CREATE TEMP TABLE s2s3_csort_freq AS
        SELECT csort, COUNT(*) as cnt FROM s2s3n WHERE length(csort) >= 4 GROUP BY csort HAVING COUNT(*) <= 100;

        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'exact_stripped_sorted' AS channel
            FROM s1n a 
            JOIN s2s3n b ON a.csort = b.csort AND a.country = b.country
            JOIN s2s3_csort_freq f ON a.csort = f.csort
            WHERE length(a.csort) >= 4 AND a.entity_id <> b.entity_id
        ) TO '{fwd6}' (DELIMITER '\t', HEADER true);

        DROP TABLE s2s3_csort_freq;
    """)
    ch_files.append(p6)

    # 7. CH7: name_prefix2 (Natural word order!)
    p7 = tmp / "ch7.tsv"
    log.info("  Generating CH7: name_prefix2 (natural order) ...")
    fwd7 = str(p7).replace("\\", "/")
    con.execute(f"""
        CREATE TEMP TABLE s2s3_p2_freq AS
        SELECT cpref2, COUNT(*) as cnt FROM s2s3n WHERE length(cpref2) >= 6 GROUP BY cpref2 HAVING COUNT(*) <= 100;

        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'name_prefix2' AS channel
            FROM s1n a 
            JOIN s2s3n b ON a.cpref2 = b.cpref2 AND a.country = b.country
            JOIN s2s3_p2_freq f ON a.cpref2 = f.cpref2
            WHERE length(a.cpref2) >= 6 AND a.entity_id <> b.entity_id
        ) TO '{fwd7}' (DELIMITER '\t', HEADER true);

        DROP TABLE s2s3_p2_freq;
    """)
    ch_files.append(p7)

    # 8. CH8: addr_street
    p8 = tmp / "ch8.tsv"
    log.info("  Generating CH8: addr_street ...")
    fwd8 = str(p8).replace("\\", "/")
    con.execute(f"""
        CREATE TEMP TABLE s2s3_addr_freq AS
        SELECT addr_street, COUNT(*) as cnt FROM s2s3n WHERE length(addr_street) >= 6 GROUP BY addr_street HAVING COUNT(*) <= 50;

        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'addr_street' AS channel
            FROM s1n a 
            JOIN s2s3n b ON a.addr_street = b.addr_street AND a.country = b.country
            JOIN s2s3_addr_freq f ON a.addr_street = f.addr_street
            WHERE length(a.addr_street) >= 6 AND a.entity_id <> b.entity_id
        ) TO '{fwd8}' (DELIMITER '\t', HEADER true);

        DROP TABLE s2s3_addr_freq;
    """)
    ch_files.append(p8)

    # 9. CH9: name_w1 (Natural first word!)
    p9 = tmp / "ch9.tsv"
    log.info("  Generating CH9: name_w1 ...")
    fwd9 = str(p9).replace("\\", "/")
    con.execute(f"""
        CREATE TEMP TABLE s2s3_w1_freq AS
        SELECT cw1, COUNT(*) as cnt FROM s2s3n WHERE length(cw1) >= 6 GROUP BY cw1 HAVING COUNT(*) <= 40;

        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'name_w1' AS channel
            FROM s1n a 
            JOIN s2s3n b ON a.cw1 = b.cw1 AND a.country = b.country
            JOIN s2s3_w1_freq f ON a.cw1 = f.cw1
            WHERE length(a.cw1) >= 6 AND a.entity_id <> b.entity_id
        ) TO '{fwd9}' (DELIMITER '\t', HEADER true);

        DROP TABLE s2s3_w1_freq;
    """)
    ch_files.append(p9)

    # 10. CH10: exact_nospace
    p10 = tmp / "ch10.tsv"
    log.info("  Generating CH10: exact_nospace ...")
    fwd10 = str(p10).replace("\\", "/")
    con.execute(f"""
        CREATE TEMP TABLE s2s3_ns_freq AS
        SELECT nospace, COUNT(*) as cnt FROM s2s3n WHERE length(nospace) >= 5 GROUP BY nospace HAVING COUNT(*) <= 50;

        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'exact_nospace' AS channel
            FROM s1n a 
            JOIN s2s3n b ON a.nospace = b.nospace AND a.country = b.country
            JOIN s2s3_ns_freq f ON a.nospace = f.nospace
            WHERE length(a.nospace) >= 5 AND a.entity_id <> b.entity_id
        ) TO '{fwd10}' (DELIMITER '\t', HEADER true);

        DROP TABLE s2s3_ns_freq;
    """)
    ch_files.append(p10)

    # 11. CH11: addr_digits
    p11 = tmp / "ch11.tsv"
    log.info("  Generating CH11: addr_digits ...")
    fwd11 = str(p11).replace("\\", "/")
    con.execute(f"""
        CREATE TEMP TABLE s2s3_digits_freq AS
        SELECT addr_nums, COUNT(*) as cnt FROM s2s3n WHERE length(addr_nums) >= 3 GROUP BY addr_nums HAVING COUNT(*) <= 50;

        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'addr_digits' AS channel
            FROM s1n a 
            JOIN s2s3n b ON a.addr_nums = b.addr_nums AND a.country = b.country
            JOIN s2s3_digits_freq f ON a.addr_nums = f.addr_nums
            WHERE length(a.addr_nums) >= 3 AND a.entity_id <> b.entity_id
        ) TO '{fwd11}' (DELIMITER '\t', HEADER true);

        DROP TABLE s2s3_digits_freq;
    """)
    ch_files.append(p11)

    # 3. CH3 & 4. CH4: Native SQL Token Unnesting (with raised 30,000 cap for 2+ tokens!)
    p3 = tmp / "ch3.tsv"
    p4 = tmp / "ch4.tsv"
    log.info("  Generating token channels CH3 & CH4 via native DuckDB SQL ...")
    fwd3 = str(p3).replace("\\", "/")
    fwd4 = str(p4).replace("\\", "/")

    con.execute(f"""
        CREATE TEMP TABLE s1_tok AS 
        SELECT entity_id, UNNEST(string_split(name_alphanum, ' ')) AS token 
        FROM s1n;

        CREATE TEMP TABLE tgt_tok AS 
        SELECT entity_id, UNNEST(string_split(name_alphanum, ' ')) AS token 
        FROM s2s3n;

        CREATE TEMP TABLE tok_freq AS 
        SELECT token, COUNT(*) as cnt 
        FROM tgt_tok 
        WHERE length(token) >= 3 
        GROUP BY token;

        CREATE TEMP TABLE tok_rare AS 
        SELECT t.entity_id, t.token 
        FROM tgt_tok t JOIN tok_freq f ON t.token = f.token 
        WHERE f.cnt <= 200;

        CREATE TEMP TABLE tok_medium AS 
        SELECT t.entity_id, t.token 
        FROM tgt_tok t JOIN tok_freq f ON t.token = f.token 
        WHERE f.cnt > 200 AND f.cnt <= 30000;

        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'token_rare' AS channel
            FROM s1_tok a JOIN tok_rare b ON a.token = b.token
            WHERE a.entity_id <> b.entity_id
        ) TO '{fwd3}' (DELIMITER '\t', HEADER true);

        COPY (
            SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'token_medium' AS channel
            FROM s1_tok a JOIN tok_medium b ON a.token = b.token
            WHERE a.entity_id <> b.entity_id
            GROUP BY a.entity_id, b.entity_id
            HAVING COUNT(*) >= 2
        ) TO '{fwd4}' (DELIMITER '\t', HEADER true);

        DROP TABLE s1_tok;
        DROP TABLE tgt_tok;
        DROP TABLE tok_freq;
        DROP TABLE tok_rare;
        DROP TABLE tok_medium;
    """)
    ch_files.append(p3)
    ch_files.append(p4)

    con.close()
    log.info("  All 11 enhanced blocking channels ready in %.1f min.", (time.time() - t0) / 60)
    return ch_files


def deduplicate_channels_to_flat(ch_files: list[pathlib.Path], out_flat: pathlib.Path) -> None:
    """Deduplicate all channel candidate pairs into a single flat TSV using DuckDB."""
    log.info("=== Deduplicating Channels to %s ===", out_flat.name)
    t0 = time.time()
    db_u = str(out_flat.parent / "_ddb_union.db")
    for f in [db_u, db_u + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    con = duckdb.connect(database=db_u)
    con.execute("SET memory_limit='8GB'")
    con.execute("SET threads=8")

    queries = []
    for p in ch_files:
        if p.exists() and p.stat().st_size > 100:
            fwd = str(p).replace("\\", "/")
            queries.append(f"SELECT s1_id, target_id, channel FROM read_csv_auto('{fwd}', delim='\t')")

    selects = " UNION ALL ".join(queries)
    fwd_out = str(out_flat).replace("\\", "/")

    con.execute(f"""
        COPY (
            SELECT s1_id, target_id, min(channel) as channel
            FROM ({selects})
            GROUP BY s1_id, target_id
        ) TO '{fwd_out}' (DELIMITER '\t', HEADER true)
    """)
    con.close()
    for f in [db_u, db_u + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    mb = out_flat.stat().st_size / 1e6
    log.info("  Deduplication complete: %s (%.1f MB) in %.1f min.", out_flat.name, mb, (time.time() - t0) / 60)


def compute_string_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Compute vectorized Rapidfuzz string features."""
    a_name = df["a_name"].fillna("").astype(str).tolist()
    b_name = df["b_name"].fillna("").astype(str).tolist()
    a_exp  = df["a_exp"].fillna("").astype(str).tolist()
    b_exp  = df["b_exp"].fillna("").astype(str).tolist()
    a_addr = df["a_addr"].fillna("").astype(str).tolist()
    b_addr = df["b_addr"].fillna("").astype(str).tolist()

    df["name_token_sort_ratio"]    = [rf_fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(a_name, b_name)]
    df["name_partial_ratio"]       = [rf_fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(a_name, b_name)]
    df["name_expanded_sort_ratio"] = [rf_fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(a_exp, b_exp)]
    df["name_qratio"]              = [rf_fuzz.QRatio(a, b) / 100.0 for a, b in zip(a_name, b_name)]
    df["name_jaro_winkler"]        = [rf_dist.JaroWinkler.similarity(a, b) for a, b in zip(a_name, b_name)]
    df["addr_jaro_winkler"]        = [rf_dist.JaroWinkler.similarity(a, b) for a, b in zip(a_addr, b_addr)]

    f_jaccard = []
    for a, b in zip(a_addr, b_addr):
        if not a or not b:
            f_jaccard.append(0.0)
        elif a == b:
            f_jaccard.append(1.0)
        else:
            sa, sb = set(a.split()), set(b.split())
            un = len(sa | sb)
            f_jaccard.append(len(sa & sb) / un if un else 0.0)
    df["addr_token_jaccard"] = f_jaccard

    f_tok_len = []
    for a, b in zip(a_name, b_name):
        la = len(a.split())
        lb = len(b.split())
        f_tok_len.append(min(la, lb) / max(la, lb, 1))
    df["name_token_len_ratio"] = f_tok_len

    df.drop(columns=["a_name", "b_name", "a_exp", "b_exp", "a_addr", "b_addr"], inplace=True, errors="ignore")
    return df


def stream_score_test_candidates(
    con: duckdb.DuckDBPyConnection,
    test_flat_path: pathlib.Path,
    test_norm_path: pathlib.Path,
    model_path: pathlib.Path,
    calibrator_path: pathlib.Path,
    threshold: float = 0.45,
) -> pd.DataFrame:
    """Stream candidate pairs, join with normalized sources, score with LightGBM, filter >= threshold."""
    log.info("=== Streaming & Scoring Test Candidates (threshold=%.2f) ===", threshold)
    t0 = time.time()

    booster = lgb.Booster(model_file=str(model_path))
    with open(calibrator_path, "rb") as f:
        iso = pickle.load(f)

    fwd_flat = str(test_flat_path).replace("\\", "/")
    fwd_norm = str(test_norm_path).replace("\\", "/")

    query = f"""
        WITH cands AS (
            SELECT s1_id, target_id 
            FROM read_csv('{fwd_flat}', delim='\t', header=true, columns={{'s1_id': 'VARCHAR', 'target_id': 'VARCHAR', 'channel': 'VARCHAR'}})
        )
        SELECT 
            c.s1_id, c.target_id,
            s1.name_alphanum AS a_name,
            tgt.name_alphanum AS b_name,
            s1.name_expanded AS a_exp,
            tgt.name_expanded AS b_exp,
            s1.addr_alphanum AS a_addr,
            tgt.addr_alphanum AS b_addr,
            (s1.country = tgt.country)::FLOAT AS country_match,
            (s1.name_sorted = tgt.name_sorted)::FLOAT AS name_sorted_exact,
            (s1.name_alphanum = tgt.name_alphanum)::FLOAT AS name_alphanum_exact,
            (s1.nospace = tgt.nospace AND length(s1.nospace) >= 5)::FLOAT AS name_nospace_exact,
            (s1.addr_nums = tgt.addr_nums AND length(s1.addr_nums) >= 3)::FLOAT AS addr_digits_exact,
            (s1.name_prefix5 = tgt.name_prefix5)::FLOAT AS name_prefix5_exact,
            (s1.addr_prefix4 = tgt.addr_prefix4)::FLOAT AS addr_prefix4_match,
            ((s1.addr_numeric = tgt.addr_numeric) AND length(s1.addr_numeric) >= 2)::FLOAT AS addr_numeric_match,
            (least(length(s1.name_alphanum), length(tgt.name_alphanum))::FLOAT / 
             greatest(length(s1.name_alphanum), length(tgt.name_alphanum), 1)::FLOAT) AS name_len_ratio
        FROM cands c
        JOIN read_parquet('{fwd_norm}') s1 ON c.s1_id = s1.entity_id
        JOIN read_parquet('{fwd_norm}') tgt ON c.target_id = tgt.entity_id
    """

    reader = con.execute(query).arrow(150000)
    passing_dfs = []
    batch_idx = 0
    total_scored = 0

    while True:
        try:
            arrow_batch = reader.read_next_batch()
        except StopIteration:
            break
        batch_idx += 1
        df_chunk = arrow_batch.to_pandas()
        if df_chunk.empty:
            continue

        n_rows = len(df_chunk)
        total_scored += n_rows

        df_feat = compute_string_features_fast(df_chunk)
        X = df_feat[FEAT_COLS].values.astype(np.float32)
        raw_probs = booster.predict(X, num_threads=8)
        cal_probs = iso.predict(raw_probs)

        mask = cal_probs >= threshold
        if np.any(mask):
            pass_df = pd.DataFrame({
                "s1_id": df_feat.loc[mask, "s1_id"].values,
                "target_id": df_feat.loc[mask, "target_id"].values,
                "cal_prob": cal_probs[mask],
            })
            passing_dfs.append(pass_df)

        if batch_idx % 20 == 0:
            n_pass = sum(len(d) for d in passing_dfs)
            log.info("  Batch %d: %d candidates scored, %d pairs >= %.2f", batch_idx, total_scored, n_pass, threshold)

    if passing_dfs:
        res_df = pd.concat(passing_dfs, ignore_index=True)
    else:
        res_df = pd.DataFrame(columns=["s1_id", "target_id", "cal_prob"])

    log.info("  Scored %d test candidate pairs in %.1f min. %d pairs passed threshold.",
             total_scored, (time.time() - t0) / 60, len(res_df))
    return res_df


def resolve_matches_and_candidates(
    con_test: duckdb.DuckDBPyConnection,
    passing_pairs: pd.DataFrame,
    test_flat_path: pathlib.Path,
    all_test_s1_ids: list[str],
    matching_out: pathlib.Path,
    cand_pairs_out: pathlib.Path,
) -> None:
    """Apply greedy target deduplication, hard capping at 15 matches, and write both compliant TSVs."""
    log.info("=== Resolving Final Matches and Candidates ===")
    t0 = time.time()

    # 1. Greedy target deduplication: each target assigned to highest-confidence S1
    if not passing_pairs.empty:
        passing_pairs = passing_pairs.sort_values("cal_prob", ascending=False)
        passing_pairs = passing_pairs.drop_duplicates(subset=["target_id"], keep="first")
        log.info("  Passing pairs after target deduplication: %d", len(passing_pairs))

    pred_map = {s: [] for s in all_test_s1_ids}
    if not passing_pairs.empty:
        passing_pairs = passing_pairs.sort_values(["s1_id", "cal_prob"], ascending=[True, False])
        for s, grp in passing_pairs.groupby("s1_id"):
            pred_map[s] = grp["target_id"].tolist()[:15]

    # Write matching_results.tsv
    matching_rows = [{"source1_entity_id": s, "matched_entity_ids": ",".join(pred_map[s])} for s in all_test_s1_ids]
    pd.DataFrame(matching_rows).to_csv(matching_out, sep="\t", index=False)
    n_singletons = sum(1 for m in pred_map.values() if len(m) == 0)
    n_links = sum(len(m) for m in pred_map.values())
    log.info("  Wrote %s: %d rows (%d singletons, %d links) in %.1f min.",
             matching_out.name, len(all_test_s1_ids), n_singletons, n_links, (time.time() - t0) / 60)

    # 2. Write candidate_pairs.tsv with match prioritization
    log.info("  Writing %s with match prioritization ...", cand_pairs_out.name)
    fwd_test_flat = str(test_flat_path).replace("\\", "/")
    
    top_cands = con_test.execute(f"""
        WITH ranked AS (
            SELECT s1_id, target_id,
                   ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY channel ASC) as rn
            FROM read_csv('{fwd_test_flat}', delim='\t', header=true, columns={{'s1_id': 'VARCHAR', 'target_id': 'VARCHAR', 'channel': 'VARCHAR'}})
        )
        SELECT s1_id, target_id
        FROM ranked
        WHERE rn <= 95
    """).df()

    cand_map = {s: list(pred_map[s]) for s in all_test_s1_ids}
    for s, grp in top_cands.groupby("s1_id"):
        for t in grp["target_id"].tolist():
            if len(cand_map[s]) >= 100:
                break
            if t not in cand_map[s]:
                cand_map[s].append(t)

    cand_rows = [{"source1_entity_id": s, "candidate_entity_ids": ",".join(cand_map[s])} for s in all_test_s1_ids]
    pd.DataFrame(cand_rows).to_csv(cand_pairs_out, sep="\t", index=False)
    log.info("  Wrote %s: %d rows.", cand_pairs_out.name, len(cand_rows))


def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("STARTING HIGH-RECALL EXPANSION RUN (TARGET: 90%+ LEADERBOARD)")
    log.info("=" * 70)

    model_path = ARTS / "lgbm_model.txt"
    cal_path   = ARTS / "calibrator.pkl"

    if not model_path.exists() or not cal_path.exists():
        log.error("Trained model or calibrator not found in %s! Exiting.", ARTS)
        sys.exit(1)

    # STEP 1: Normalize test sources with French suffixes, accent folding, and natural prefix order
    log.info("\n[STEP 1] Generating Enhanced test_sources_norm.parquet ...")
    test_norm_cache = ARTS / "test_sources_norm.parquet"
    
    # We remove the old parquet to force full enhanced normalization
    if test_norm_cache.exists():
        test_norm_cache.unlink()

    ts1 = pd.read_csv(TEST_DIR / "test_source1.tsv", sep="\t", dtype=str)
    ts2 = pd.read_csv(TEST_DIR / "test_source2.tsv", sep="\t", dtype=str)
    ts3 = pd.read_csv(TEST_DIR / "test_source3.tsv", sep="\t", dtype=str)
    all_test_s1_ids = ts1["entity_id"].tolist()

    ts1n = normalize_sources_v4(ts1)
    ts2n = normalize_sources_v4(ts2)
    ts3n = normalize_sources_v4(ts3)
    ts2s3n = pd.concat([ts2n, ts3n], ignore_index=True)
    del ts2, ts3, ts2n, ts3n
    all_test_norm = pd.concat([ts1n, ts2s3n], ignore_index=True)
    all_test_norm.to_parquet(test_norm_cache, index=False)
    del ts1, ts1n, ts2s3n, all_test_norm
    gc.collect()

    # STEP 2: Enhanced 11-Channel Blocking
    log.info("\n[STEP 2] Running Enhanced 11-Channel Blocking on Test Set ...")
    TMP_TEST = ARTS / "_tmp_test_channels"
    if TMP_TEST.exists():
        shutil.rmtree(TMP_TEST, ignore_errors=True)
    TMP_TEST.mkdir(parents=True, exist_ok=True)

    test_flat = ARTS / "test_candidates_flat.tsv"
    if test_flat.exists():
        test_flat.unlink()

    test_ch_files = build_enhanced_blocking_channels(test_norm_cache, TMP_TEST)
    deduplicate_channels_to_flat(test_ch_files, test_flat)

    # STEP 3: Stream and Score Test Candidates with threshold 0.45
    log.info("\n[STEP 3] Streaming & Scoring Test Candidates (threshold 0.45) ...")
    con_test = duckdb.connect()
    con_test.execute("SET memory_limit='8GB'")
    con_test.execute("SET threads=8")

    passing_pairs = stream_score_test_candidates(
        con_test,
        test_flat,
        test_norm_cache,
        model_path,
        cal_path,
        threshold=0.45,
    )
    con_test.close()
    gc.collect()

    # STEP 4: Format and Output TSVs
    log.info("\n[STEP 4] Formatting Final Output TSVs ...")
    matching_results_path = OUT_DIR / "matching_results.tsv"
    cand_pairs_path = OUT_DIR / "candidate_pairs.tsv"
    con_resolve = duckdb.connect()
    con_resolve.execute("SET memory_limit='8GB'")
    con_resolve.execute("SET threads=8")
    resolve_matches_and_candidates(
        con_resolve,
        passing_pairs,
        test_flat,
        all_test_s1_ids,
        matching_results_path,
        cand_pairs_path,
    )
    con_resolve.close()

    # STEP 5: Official Validation Check
    log.info("\n[STEP 5] Running Official Submission Validator ...")
    val_script = REPO / "utils" / "validate_submission.py"
    if val_script.exists():
        val_cmd = [
            sys.executable, str(val_script),
            "--matching", str(matching_results_path),
            "--candidate", str(cand_pairs_path),
            "--test-dir", str(TEST_DIR),
        ]
        res = subprocess.run(val_cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.returncode != 0:
            log.error("VALIDATION FAILED: %s", res.stderr)
        else:
            log.info(">>> VALIDATION PASSED! <<<")

    # STEP 6: Package into ZIP
    log.info("\n[STEP 6] Updating Dunder_submission.zip ...")
    pkg_script = REPO / "package_submission.py"
    if pkg_script.exists():
        pkg_cmd = [sys.executable, str(pkg_script), "--team-name", "Dunder"]
        res_pkg = subprocess.run(pkg_cmd, capture_output=True, text=True)
        print(res_pkg.stdout)

    log.info("\n" + "=" * 70)
    log.info("PIPELINE FULLY COMPLETE IN %.1f MINUTES!", (time.time() - t_start) / 60)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
