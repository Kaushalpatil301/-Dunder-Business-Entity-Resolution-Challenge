#!/usr/bin/env python
"""
run_after_blocking.py — Run Phases 5-16 assuming flat candidates already exist.

Use this after add_new_channels.py completes successfully.
Skips blocking; runs feature extraction, training, calibration, test inference, validation.

Run from repo root:
    python run_after_blocking.py

To also re-run test blocking:
    python run_after_blocking.py --redo-test-block
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

def _ensure(pkg, name=None):
    import importlib.util
    if importlib.util.find_spec(name or pkg) is None:
        print(f"[setup] Installing {pkg} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

_ensure("pyyaml",    "yaml")
_ensure("lightgbm",  "lightgbm")
_ensure("rapidfuzz", "rapidfuzz")
_ensure("duckdb",    "duckdb")
_ensure("sparse-dot-topn", "sparse_dot_topn")
_ensure("scipy",     "scipy")

import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
from rapidfuzz import fuzz as rf_fuzz
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
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
log = logging.getLogger("pipeline")

from config.config import load_config

CFG       = load_config()
ARTS      = REPO / CFG["paths"]["artifacts_dir"]
TRAIN_DIR = REPO / CFG["paths"]["train_dir"]
TEST_DIR  = REPO / CFG["paths"]["test_dir"]
OUT_DIR   = REPO / CFG["paths"]["output_dir"]
ARTS.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

FEAT_COLS = [
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_expanded_sort_ratio",
    "name_qratio",
    "name_jaro_winkler",
    "name_sorted_exact",
    "name_alphanum_exact",
    "addr_token_jaccard",
    "addr_numeric_match",
    "addr_jaro_winkler",
    "addr_prefix4_match",
    "country_match",
    "name_len_ratio",
    "name_token_len_ratio",
    "name_prefix5_exact",
]

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

    name_prefix5 = sorted_view.str[:5]

    addr = df["business_address"].fillna("").str.lower().str.strip()
    addr_alnum = addr.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    addr_alnum = addr_alnum.str.replace(r"\s+", " ", regex=True).str.strip()
    addr_num   = addr_alnum.str.extract(r"(\d+)", expand=False).fillna("")
    addr_prefix4 = addr_alnum.str[:4]

    out = df[["entity_id", "business_name", "business_address", "country"]].copy()
    out["name_alphanum"]        = alnum
    out["name_sorted"]          = sorted_view
    out["name_expanded"]        = exp
    out["name_expanded_sorted"] = exp_sorted
    out["name_prefix5"]         = name_prefix5
    out["addr_alphanum"]        = addr_alnum
    out["addr_numeric"]         = addr_num
    out["addr_prefix4"]         = addr_prefix4
    return out


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    u = len(ta | tb)
    return len(ta & tb) / u if u else 0.0


def extract_features(
    candidates_flat: pd.DataFrame,
    lookup: pd.DataFrame,
    gt_lookup,
    out_path: pathlib.Path,
    chunk: int = 150_000,
) -> None:
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

        feat["country_match"] = (s1v["country"].values == tgtv["country"].values).astype("float32")

        a_len = np.maximum(s1v["name_alphanum"].str.len().values, 1)
        b_len = np.maximum(tgtv["name_alphanum"].str.len().values, 1)
        feat["name_len_ratio"] = (np.minimum(a_len, b_len) / np.maximum(a_len, b_len)).astype("float32")

        a_tok = np.maximum(s1v["name_alphanum"].str.split().str.len().values, 1)
        b_tok = np.maximum(tgtv["name_alphanum"].str.split().str.len().values, 1)
        feat["name_token_len_ratio"] = (np.minimum(a_tok, b_tok) / np.maximum(a_tok, b_tok)).astype("float32")

        feat["name_prefix5_exact"] = (s1v["name_prefix5"].values == tgtv["name_prefix5"].values).astype("float32")

        if gt_lookup is not None:
            feat["label"] = [
                1 if slc.loc[i, "target_id"] in gt_lookup.get(slc.loc[i, "s1_id"], frozenset())
                else 0
                for i in slc.index
            ]

        feat.to_csv(out_path, sep="\t", index=False,
                    header=first, mode="w" if first else "a", encoding="utf-8")
        first = False
        if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
            log.info("    chunk %d/%d done", ci + 1, n_chunks)

    log.info("  Done: %s", out_path)


def _f05(probs, labels, t):
    preds = (probs >= t).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    d  = 0.25 * p + r
    return 1.25 * p * r / d if d > 0 else 0.0


def train_model(feat_a, feat_b, model_out, threshold_out):
    log.info("=== PHASE 6: LightGBM Training ===")

    def _load(p):
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

    model = lgb.LGBMClassifier(
        objective="binary", boosting_type="gbdt",
        num_leaves=127,
        learning_rate=0.03,
        n_estimators=1000,
        min_child_samples=20,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=spw,
        random_state=SEED, n_jobs=-1, verbose=-1,
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

    va_probs = model.predict_proba(X_va)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.01)
    scores = [(_f05(va_probs, y_va, t), t) for t in thresholds]
    best_f, best_t = max(scores, key=lambda x: x[0])
    log.info("  Best threshold: %.3f  → pairwise F0.5=%.4f", best_t, best_f)
    threshold_out.write_text(f"{best_t:.4f}\n", encoding="utf-8")
    return best_t


def fit_calibrator(feat_c, model_path, cal_out):
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


def predict_and_calibrate(feat_path, model_path, cal_path, out_path, chunk=500_000):
    booster = lgb.Booster(model_file=str(model_path))
    with open(cal_path, "rb") as f:
        iso = pickle.load(f)

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


def resolve(scores_path, all_s1_ids, threshold, output_path, chunk=1_000_000):
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

    # Target-side dedup (greedy)
    matched = matched.sort_values("cal_prob", ascending=False).drop_duplicates(
        subset=["target_id"], keep="first")
    log.info("  After target dedup: %d pairs", len(matched))

    result = {s: [] for s in all_s1_ids}
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
    assert len(out_df) == len(all_s1_ids), f"ID set mismatch: {len(out_df)} vs {len(all_s1_ids)}"
    log.info("  Gate PASSED: all %d test S1 IDs present in output.", len(all_s1_ids))


def _addr_key(s):
    ex = s.str.extract(r"(?<!\d)(\d{4,})\s+([a-z]{4,})", expand=True)
    return (ex[0].fillna("") + " " + ex[1].fillna("")).str.strip()


def _tfidf_block(s1n, s2s3n, out_path, field, top_n, threshold, channel_name,
                  ngram_range=(2, 3), chunk_size=25_000):
    s1_texts   = s1n[field].fillna("").tolist()
    s2s3_texts = s2s3n[field].fillna("").tolist()
    s1_ids     = s1n["entity_id"].tolist()
    s2s3_ids   = s2s3n["entity_id"].tolist()
    corpus     = s1_texts + s2s3_texts
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                          min_df=2, max_df=0.95, sublinear_tf=True)
    vec.fit(corpus)
    s2s3_mat = vec.transform(s2s3_texts)
    s1_mat   = vec.transform(s1_texts)
    n_cpu = max(1, (os.cpu_count() or 4) // 2)
    first = True; total = 0
    for start in range(0, len(s1_ids), chunk_size):
        end = min(start + chunk_size, len(s1_ids))
        cx  = awesome_cossim_topn(s1_mat[start:end], s2s3_mat.T,
                                   ntop=top_n, lower_bound=threshold,
                                   use_threads=True, n_threads=n_cpu).tocoo()
        if cx.nnz == 0: continue
        pairs = pd.DataFrame({
            "s1_id":     [s1_ids[start + r] for r in cx.row],
            "target_id": [s2s3_ids[c]        for c in cx.col],
            "channel":   channel_name,
        })
        pairs = pairs[pairs["s1_id"] != pairs["target_id"]]
        pairs.to_csv(out_path, sep="\t", index=False,
                     header=first, mode="w" if first else "a", encoding="utf-8")
        first = False; total += len(pairs)
    if first:
        pd.DataFrame(columns=["s1_id", "target_id", "channel"]).to_csv(
            out_path, sep="\t", index=False, encoding="utf-8")
    log.info("    [%s]: %d pairs", channel_name, total)


def block_test(s1n, s2s3n, flat_out, grouped_out):
    log.info("=== PHASE 15a: Test Blocking (9-channel) ===")
    t0  = time.time()
    tmp = ARTS / "_tmp_test_channels"
    tmp.mkdir(parents=True, exist_ok=True)
    n_cpu = os.cpu_count() or 4

    ch_files = []

    # DuckDB channels
    db = str(tmp / "_ddb_test.db")
    for f in [db, db + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    s1n["addr_key"]   = _addr_key(s1n["addr_alphanum"])
    s2s3n["addr_key"] = _addr_key(s2s3n["addr_alphanum"])

    con = duckdb.connect(database=db)
    con.execute("SET memory_limit='12GB'")
    con.execute(f"SET threads={max(4, n_cpu // 2)}")
    con.execute("SET preserve_insertion_order=false")
    con.register("s1n", s1n); con.register("s2s3n", s2s3n)

    for ch_name, sql, path in [
        ("ch1", """SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'exact_sorted' AS channel
                   FROM s1n a JOIN s2s3n b ON a.name_sorted=b.name_sorted
                   WHERE length(a.name_sorted)>=3 AND a.entity_id<>b.entity_id""",
         tmp / "ch1.tsv"),
        ("ch2", """SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'exact_expanded_sorted' AS channel
                   FROM s1n a JOIN s2s3n b ON a.name_expanded_sorted=b.name_expanded_sorted
                   WHERE length(a.name_expanded_sorted)>=3 AND a.entity_id<>b.entity_id""",
         tmp / "ch2.tsv"),
        ("ch5", """SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'addr_composite' AS channel
                   FROM s1n a JOIN s2s3n b ON a.addr_key=b.addr_key AND a.country=b.country
                   WHERE length(a.addr_key)>=8 AND a.entity_id<>b.entity_id""",
         tmp / "ch5.tsv"),
        ("ch6", """SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'addr_numeric_country' AS channel
                   FROM s1n a JOIN s2s3n b ON a.addr_numeric=b.addr_numeric AND a.country=b.country
                   WHERE length(a.addr_numeric)>=4 AND a.entity_id<>b.entity_id""",
         tmp / "ch6.tsv"),
    ]:
        log.info("  Test %s ...", ch_name)
        fwd = str(path).replace("\\", "/")
        con.execute(f"COPY ({sql}) TO '{fwd}' (DELIMITER '\t', HEADER true)")
        ch_files.append(path)
        log.info("    done (%.1f min)", (time.time() - t0) / 60)

    con.close()
    for f in [db, db + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

    # Token channels (CH3, CH4)
    log.info("  Test token channels (CH3, CH4) ...")
    MIN_TOK_LEN = 3; MAX_RARE_FREQ = 200; MAX_MED_FREQ = 5000; MIN_MED_SH = 2

    s23_tok = (s2s3n[["entity_id", "name_alphanum"]]
               .assign(token=s2s3n["name_alphanum"].str.split()).explode("token"))
    s23_tok = s23_tok[s23_tok["token"].str.len() >= MIN_TOK_LEN].copy()
    freq = s23_tok.groupby("token")["entity_id"].nunique()
    tok_rare   = s23_tok[s23_tok["token"].isin(freq[freq <= MAX_RARE_FREQ].index)][["token", "entity_id"]].drop_duplicates()
    tok_medium = s23_tok[s23_tok["token"].isin(
        freq[(freq > MAX_RARE_FREQ) & (freq <= MAX_MED_FREQ)].index
    )][["token", "entity_id"]].drop_duplicates()
    del s23_tok

    s1_tok = (s1n[["entity_id", "name_alphanum"]]
              .assign(token=s1n["name_alphanum"].str.split()).explode("token"))
    s1_tok = s1_tok[s1_tok["token"].str.len() >= MIN_TOK_LEN].copy()

    # CH3
    db3 = str(tmp / "_ddb3.db")
    for f in [db3, db3 + ".wal"]:
        if os.path.exists(f): os.remove(f)
    con3 = duckdb.connect(database=db3)
    con3.execute("SET memory_limit='12GB'"); con3.execute(f"SET threads={n_cpu}")
    con3.execute("SET preserve_insertion_order=false")
    con3.register("s1_tok", s1_tok); con3.register("tok_rare", tok_rare)
    p3 = tmp / "ch3.tsv"
    fwd3 = str(p3).replace("\\", "/")
    con3.execute(f"""
        COPY (SELECT DISTINCT s.entity_id AS s1_id, t.entity_id AS target_id, 'token_rare' AS channel
              FROM s1_tok s JOIN tok_rare t ON s.token=t.token
              WHERE s.entity_id<>t.entity_id) TO '{fwd3}' (DELIMITER '\t', HEADER true)""")
    con3.close()
    for f in [db3, db3 + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
    ch_files.append(p3); del tok_rare

    # CH4
    s1_ids = s1_tok["entity_id"].drop_duplicates().values
    n_chunks = 4; cz = int(np.ceil(len(s1_ids) / n_chunks)); ch4_parts = []
    for i in range(n_chunks):
        sub = set(s1_ids[i * cz:(i + 1) * cz])
        s1_sub = s1_tok[s1_tok["entity_id"].isin(sub)]
        db4 = str(tmp / f"_ddb4_{i}.db")
        for f in [db4, db4 + ".wal"]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        con4 = duckdb.connect(database=db4)
        con4.execute("SET memory_limit='10GB'"); con4.execute("SET threads=4")
        con4.execute("SET preserve_insertion_order=false")
        con4.register("s1_tok_sub", s1_sub); con4.register("tok_medium", tok_medium)
        p4p = tmp / f"ch4_part_{i}.tsv"
        fwd4 = str(p4p).replace("\\", "/")
        con4.execute(f"""
            COPY (SELECT s.entity_id AS s1_id, t.entity_id AS target_id, 'token_medium' AS channel
                  FROM s1_tok_sub s JOIN tok_medium t ON s.token=t.token
                  WHERE s.entity_id<>t.entity_id GROUP BY s.entity_id, t.entity_id
                  HAVING COUNT(*)>={MIN_MED_SH}) TO '{fwd4}' (DELIMITER '\t', HEADER true)""")
        con4.close()
        for f in [db4, db4 + ".wal"]:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass
        ch4_parts.append(p4p)

    p4 = tmp / "ch4.tsv"
    with open(p4, "w", encoding="utf-8") as f_out:
        first = True
        for pp in ch4_parts:
            if not pp.exists(): continue
            with open(pp, "r", encoding="utf-8") as f_in:
                hdr = f_in.readline()
                if first: f_out.write(hdr); first = False
                for line in f_in: f_out.write(line)
            try: os.remove(pp)
            except: pass
    ch_files.append(p4); del tok_medium, s1_tok
    log.info("    token channels done (%.1f min)", (time.time() - t0) / 60)

    # TF-IDF channels
    log.info("  Test TF-IDF channels ...")
    p7 = tmp / "ch7.tsv"
    _tfidf_block(s1n, s2s3n, p7, "name_expanded_sorted",
                 top_n=20, threshold=0.15, channel_name="tfidf_name",
                 chunk_size=20_000)
    ch_files.append(p7)

    p8 = tmp / "ch8.tsv"
    _tfidf_block(s1n, s2s3n, p8, "name_alphanum",
                 top_n=15, threshold=0.20, channel_name="tfidf_name_word",
                 chunk_size=20_000)
    ch_files.append(p8)

    # Dedup union
    log.info("  Deduplicating test union ...")
    existing = [p for p in ch_files if p.exists() and p.stat().st_size > 50]
    dbu = str(tmp / "_ddb_union.db")
    for f in [dbu, dbu + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
    con = duckdb.connect(database=dbu)
    con.execute("SET memory_limit='14GB'")
    con.execute(f"SET threads={n_cpu}")
    selects = " UNION ALL ".join(
        [f"SELECT s1_id, target_id, channel FROM read_csv_auto('{str(p).replace(chr(92), '/')}', delim='\t', header=true)"
         for p in existing])
    union = con.execute(f"SELECT DISTINCT s1_id, target_id, channel FROM ({selects})").df()
    con.close()
    for f in [dbu, dbu + ".wal"]:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass
    log.info("  Test unique candidates: %d", len(union))

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
    log.info("  candidate_pairs.tsv: %d rows (%.1f min)", len(grouped), (time.time() - t0) / 60)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo-test-block", action="store_true",
                        help="Re-run test blocking even if test_candidates_flat.tsv exists")
    args = parser.parse_args()

    T_START = time.time()
    log.info("=" * 70)
    log.info("PIPELINE: Phases 5-16 (post-blocking)")
    log.info("=" * 70)

    # Check flat candidates exist
    flat_train = ARTS / "candidate_pairs_all_folds_flat.tsv"
    if not flat_train.exists():
        log.error("Flat candidates not found: %s", flat_train)
        log.error("Run add_new_channels.py first.")
        sys.exit(1)

    # Load split
    split_df = pd.read_csv(ARTS / "entity_split.tsv", sep="\t", dtype=str)
    folds = {fold: set(split_df[split_df["fold"] == fold]["source1_entity_id"])
             for fold in ("A", "B", "C", "D")}
    log.info("Split: A=%d  B=%d  C=%d  D=%d",
             len(folds["A"]), len(folds["B"]), len(folds["C"]), len(folds["D"]))

    # Load & normalize train
    log.info("\n[NORMALIZE] Training sources ...")
    s1  = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2  = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3  = pd.read_csv(TRAIN_DIR / "train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt  = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

    s1n   = _norm(s1)
    s2n   = _norm(s2)
    s3n   = _norm(s3)
    s2s3n = pd.concat([s2n, s3n], ignore_index=True)
    lookup = pd.concat([s1n, s2s3n], ignore_index=True).set_index("entity_id")

    # GT lookup
    gt_lookup = {}
    for _, row in gt.iterrows():
        raw = row["matched_entity_ids"].strip()
        gt_lookup[row["source1_entity_id"]] = frozenset(raw.split(",")) if raw else frozenset()

    # Phase 5: Feature extraction
    log.info("\n=== PHASE 5: Feature Extraction ===")
    cands_all = pd.read_csv(flat_train, sep="\t", dtype=str, keep_default_na=False)
    if "source1_entity_id" in cands_all.columns:
        cands_all = cands_all.rename(
            columns={"source1_entity_id": "s1_id", "candidate_entity_id": "target_id"})

    for fold_name in ("A", "B", "C"):
        fold_ids   = folds[fold_name]
        fold_cands = cands_all[cands_all["s1_id"].isin(fold_ids)].reset_index(drop=True)
        feat_path  = ARTS / f"features_fold_{fold_name.lower()}.tsv"
        log.info("  Fold %s: %d candidate pairs → %s", fold_name, len(fold_cands), feat_path)
        if feat_path.exists():
            log.info("  (exists, skipping)")
            continue
        extract_features(fold_cands, lookup, gt_lookup, feat_path)

    # Phase 6: LightGBM
    model_path     = ARTS / "lgbm_model.txt"
    threshold_path = ARTS / "threshold.txt"
    best_t = train_model(
        ARTS / "features_fold_a.tsv",
        ARTS / "features_fold_b.tsv",
        model_path, threshold_path,
    )

    # Phase 7: Calibration
    cal_path = ARTS / "calibrator.pkl"
    fit_calibrator(ARTS / "features_fold_c.tsv", model_path, cal_path)

    elapsed = (time.time() - T_START) / 60
    log.info("Training done (%.1f min). Starting test inference ...", elapsed)

    # Phase 15: Test blocking + inference
    log.info("\n[NORMALIZE] Test sources ...")
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

    if args.redo_test_block or not test_flat_cands.exists():
        block_test(ts1n, ts2s3n, test_flat_cands, test_cands_tsv)
    else:
        log.info("  [SKIP] Test blocking — using existing %s", test_flat_cands)
        union_test = pd.read_csv(test_flat_cands, sep="\t", dtype=str, keep_default_na=False)
        union_test = union_test.rename(
            columns={"source1_entity_id": "s1_id", "candidate_entity_id": "target_id"})
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

    # Feature extraction for test
    test_lookup   = pd.concat([ts1n, ts2s3n], ignore_index=True).set_index("entity_id")
    test_cands_df = pd.read_csv(test_flat_cands, sep="\t", dtype=str, keep_default_na=False)
    test_cands_df = test_cands_df.rename(
        columns={"source1_entity_id": "s1_id", "candidate_entity_id": "target_id"})
    feat_test = ARTS / "features_test.tsv"
    extract_features(test_cands_df, test_lookup, None, feat_test)

    # Predict + calibrate
    cal_scores_test = ARTS / "cal_scores_test.tsv"
    predict_and_calibrate(feat_test, model_path, cal_path, cal_scores_test)

    # Decision
    all_test_s1      = set(ts1["entity_id"].tolist())
    matching_results = OUT_DIR / "matching_results.tsv"
    resolve(cal_scores_test, all_test_s1, best_t, matching_results)

    # Phase 16: Validate
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
        print(result.stderr, file=sys.stderr)
    if result.returncode == 0:
        log.info("PHASE 16 GATE: PASS — submission files are valid.")
    else:
        log.error("PHASE 16 GATE: FAIL — fix issues above before submitting.")
        sys.exit(1)

    elapsed = (time.time() - T_START) / 60
    log.info("\n" + "=" * 70)
    log.info("PIPELINE COMPLETE in %.1f minutes", elapsed)
    log.info("  output/candidate_pairs.tsv")
    log.info("  output/matching_results.tsv")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
