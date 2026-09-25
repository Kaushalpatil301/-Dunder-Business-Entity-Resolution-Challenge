#!/usr/bin/env python
"""
master_run.py — Single end-to-end pipeline runner (Option A Fast-Track).

Runs every phase in sequence, gates checked automatically:
  Phase 4 : Blocking union via DuckDB (all S1 entities + fold-D recall gate)
  Phase 5 : Pairwise feature extraction (RapidFuzz, chunked)
  Phase 6 : LightGBM training (fold A) + F0.5 threshold tuning (fold B)
  Phase 7 : Isotonic calibration (fold C)
  Phase 8 : Entity-level decision (test set)
  Phase 15: Full test-set inference → output/candidate_pairs.tsv + matching_results.tsv
  Phase 16: validate_submission.py → must exit 0

Run from repo root:
    python master_run.py

All artifacts saved under artifacts/.
Final outputs: output/candidate_pairs.tsv, output/matching_results.tsv
"""

from __future__ import annotations

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
    import importlib
    import importlib.util
    name = import_name or pkg
    if importlib.util.find_spec(name) is None:
        print(f"[setup] Installing {pkg} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

_ensure("pyyaml",    "yaml")
_ensure("lightgbm",  "lightgbm")
_ensure("rapidfuzz", "rapidfuzz")
_ensure("duckdb",    "duckdb")

import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
from rapidfuzz import fuzz as rf_fuzz
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

from config.config import load_config

# ─── logging ───────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
    ],
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

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

# Blocking constants
MIN_KEY_LEN   = 3
MIN_TOK_LEN   = 3
MAX_RARE_FREQ = 150
MAX_MED_FREQ  = 3_000
MIN_RARE_SH   = 1
MIN_MED_SH    = 2

# LightGBM features
FEAT_COLS = [
    "name_token_sort_ratio", "name_partial_ratio", "name_expanded_sort_ratio",
    "name_sorted_exact", "name_alphanum_exact",
    "addr_token_jaccard", "addr_numeric_match", "country_match",
    "name_len_ratio", "name_token_len_ratio",
]

# ═══════════════════════════════════════════════════════════════════════════
# STEP 0 — Normalize
# ═══════════════════════════════════════════════════════════════════════════

def _norm(df: pd.DataFrame) -> pd.DataFrame:
    """Fast in-place normalization producing all blocking views."""
    name = df["business_name"].fillna("").str.lower().str.strip()
    name = name.str.replace(r"<null>", "", regex=True)
    name = name.str.replace(r"^(smt|sri|--|[*]+)\s+", "", regex=True)
    name = name.str.replace(r"\s+", " ", regex=True).str.strip()

    alnum = name.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    alnum = alnum.str.replace(r"\s+", " ", regex=True).str.strip()

    sorted_view = alnum.str.split().apply(
        lambda t: " ".join(sorted(t)) if t else ""
    )

    exp = alnum.copy()
    for long, short in [("private","pvt"),("limited","ltd"),("incorporated","inc"),
                        ("corporation","corp"),("services","svcs"),("service","svc"),
                        ("management","mgmt"),("international","intl"),("associates","assoc")]:
        exp = exp.str.replace(rf"\b{long}\b", short, regex=True)
    for short, long in [("pvt","private"),("ltd","limited"),("inc","incorporated"),
                        ("corp","corporation"),("svcs","services"),("svc","service"),
                        ("mgmt","management"),("intl","international"),("assoc","associates")]:
        exp = exp.str.replace(rf"\b{short}\b", long, regex=True)
    exp = exp.str.replace(r"\s+", " ", regex=True).str.strip()
    exp_sorted = exp.str.split().apply(lambda t: " ".join(sorted(t)) if t else "")

    addr = df["business_address"].fillna("").str.lower().str.strip()
    addr_alnum = addr.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    addr_alnum = addr_alnum.str.replace(r"\s+", " ", regex=True).str.strip()
    addr_num   = addr_alnum.str.extract(r"(\d+)", expand=False).fillna("")

    out = df[["entity_id", "business_name", "country"]].copy()
    out["name_lower"]         = name
    out["name_alphanum"]      = alnum
    out["name_sorted"]        = sorted_view
    out["name_expanded"]      = exp
    out["name_expanded_sorted"] = exp_sorted
    out["addr_alphanum"]      = addr_alnum
    out["addr_numeric"]       = addr_num
    return out


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Blocking via DuckDB (all folds)
# ═══════════════════════════════════════════════════════════════════════════

def phase4_blocking(
    s1n: pd.DataFrame,
    s2s3n: pd.DataFrame,
    gt: pd.DataFrame,
    split_df: pd.DataFrame,
    out_flat: pathlib.Path,
) -> dict:
    """
    Run 5-channel blocking using DuckDB (parallel hash joins).
    Returns per-channel recall stats for fold D.
    Saves flat candidate pairs (s1_id, target_id, channel) to out_flat.
    """
    log.info("=== PHASE 4: DuckDB Blocking (all folds) ===")
    t0 = time.time()

    con = duckdb.connect(database=":memory:", config={"threads": os.cpu_count() or 14})

    con.register("s1n",   s1n)
    con.register("s2s3n", s2s3n)

    all_pairs: list[pd.DataFrame] = []

    # ----- CH1: exact_sorted -----
    log.info("  CH1: exact_sorted ...")
    ch1 = con.execute("""
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'exact_sorted' AS channel
        FROM s1n a
        JOIN s2s3n b ON a.name_sorted = b.name_sorted
        WHERE length(a.name_sorted) >= 3
          AND a.entity_id <> b.entity_id
    """).df()
    log.info("     %d pairs", len(ch1))
    all_pairs.append(ch1)

    # ----- CH2: exact_expanded_sorted -----
    log.info("  CH2: exact_expanded_sorted ...")
    ch2 = con.execute("""
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'exact_expanded_sorted' AS channel
        FROM s1n a
        JOIN s2s3n b ON a.name_expanded_sorted = b.name_expanded_sorted
        WHERE length(a.name_expanded_sorted) >= 3
          AND a.entity_id <> b.entity_id
    """).df()
    log.info("     %d pairs", len(ch2))
    all_pairs.append(ch2)

    # ----- CH3 & CH4: token inverted index -----
    log.info("  Building token tables ...")
    # Explode S2+S3 tokens
    s23_tok = (
        s2s3n[["entity_id", "name_alphanum"]]
        .assign(token=s2s3n["name_alphanum"].str.split())
        .explode("token")
    )
    s23_tok = s23_tok[s23_tok["token"].str.len() >= MIN_TOK_LEN].copy()
    freq = s23_tok.groupby("token")["entity_id"].nunique()

    tok_rare   = s23_tok[s23_tok["token"].isin(freq[freq <= MAX_RARE_FREQ].index)][["token","entity_id"]].drop_duplicates()
    tok_medium = s23_tok[s23_tok["token"].isin(freq[freq <= MAX_MED_FREQ].index)][["token","entity_id"]].drop_duplicates()
    log.info("  Token tables: rare=%d rows, medium=%d rows", len(tok_rare), len(tok_medium))

    # S1 tokens
    s1_tok = (
        s1n[["entity_id", "name_alphanum"]]
        .assign(token=s1n["name_alphanum"].str.split())
        .explode("token")
    )
    s1_tok = s1_tok[s1_tok["token"].str.len() >= MIN_TOK_LEN].copy()

    con.register("s1_tok",    s1_tok)
    con.register("tok_rare",  tok_rare)
    con.register("tok_medium", tok_medium)

    log.info("  CH3: token_rare (min_shared=1) ...")
    ch3 = con.execute("""
        SELECT DISTINCT s.entity_id AS s1_id, t.entity_id AS target_id, 'token_rare' AS channel
        FROM s1_tok s
        JOIN tok_rare t ON s.token = t.token
        WHERE s.entity_id <> t.entity_id
    """).df()
    log.info("     %d pairs", len(ch3))
    all_pairs.append(ch3)

    log.info("  CH4: token_medium (min_shared=2) ...")
    ch4 = con.execute("""
        SELECT s.entity_id AS s1_id, t.entity_id AS target_id,
               'token_medium' AS channel
        FROM s1_tok s
        JOIN tok_medium t ON s.token = t.token
        WHERE s.entity_id <> t.entity_id
        GROUP BY s.entity_id, t.entity_id
        HAVING COUNT(*) >= 2
    """).df()
    log.info("     %d pairs", len(ch4))
    all_pairs.append(ch4)

    # ----- CH5: addr_composite -----
    log.info("  CH5: addr_composite ...")
    def _addr_key(s: pd.Series) -> pd.Series:
        ex = s.str.extract(r"(?<!\d)(\d{4,})\s+([a-z]{4,})", expand=True)
        return (ex[0].fillna("") + " " + ex[1].fillna("")).str.strip()

    s1n["addr_key"]   = _addr_key(s1n["addr_alphanum"])
    s2s3n["addr_key"] = _addr_key(s2s3n["addr_alphanum"])
    con.register("s1n",   s1n)
    con.register("s2s3n", s2s3n)
    ch5 = con.execute("""
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'addr_composite' AS channel
        FROM s1n a
        JOIN s2s3n b ON a.addr_key = b.addr_key AND a.country = b.country
        WHERE length(a.addr_key) >= 8
          AND a.entity_id <> b.entity_id
    """).df()
    log.info("     %d pairs", len(ch5))
    all_pairs.append(ch5)

    # ----- Union & dedup -----
    log.info("  Deduplicating union ...")
    union = pd.concat(all_pairs, ignore_index=True)
    union = union.drop_duplicates(subset=["s1_id", "target_id"], keep="first")
    log.info("  Total unique candidates: %d  (%.1f min)", len(union), (time.time()-t0)/60)

    # ----- Recall gate on fold D -----
    fold_d = set(split_df[split_df["fold"] == "D"]["source1_entity_id"])
    gt_exploded = (
        gt[gt["matched_entity_ids"].str.strip() != ""]
        .assign(tgt=gt["matched_entity_ids"].str.split(","))
        .explode("tgt")
        [["source1_entity_id", "tgt"]]
        .rename(columns={"source1_entity_id": "s1_id", "tgt": "target_id"})
    )
    tp_d    = set(zip(gt_exploded[gt_exploded["s1_id"].isin(fold_d)]["s1_id"],
                      gt_exploded[gt_exploded["s1_id"].isin(fold_d)]["target_id"]))
    cand_d  = set(zip(union[union["s1_id"].isin(fold_d)]["s1_id"],
                      union[union["s1_id"].isin(fold_d)]["target_id"]))
    n_true  = len(tp_d)
    n_found = len(tp_d & cand_d)
    recall  = n_found / n_true if n_true > 0 else 0.0
    gate    = "PASS" if recall >= 0.95 else "FAIL"
    log.info("  PHASE 4 GATE [fold D]: recall=%.4f  %s  (%d/%d)", recall, gate, n_found, n_true)

    # Per-channel recall
    per_ch: dict[str, float] = {}
    for ch, grp in union[union["s1_id"].isin(fold_d)].groupby("channel"):
        ch_set = set(zip(grp["s1_id"], grp["target_id"]))
        per_ch[ch] = len(tp_d & ch_set) / n_true if n_true > 0 else 0.0
        log.info("    %-28s %.4f", ch, per_ch[ch])

    # Save report
    report_lines = [
        "PHASE 4 RECALL REPORT",
        f"  Overall recall (fold D): {recall:.4f}  [{gate}]",
        f"  True pairs:  {n_true:,}",
        f"  Found pairs: {n_found:,}",
        f"  Total candidates (all folds): {len(union):,}",
    ] + [f"  {ch}: {r:.4f}" for ch, r in sorted(per_ch.items(), key=lambda x: -x[1])]
    (ARTS / "phase4_recall_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    # ----- Save flat candidates -----
    union.to_csv(out_flat, sep="\t", index=False, encoding="utf-8")
    log.info("  Flat candidates saved: %s", out_flat)
    con.close()

    return {"recall": recall, "gate": gate, "n_true": n_true, "n_found": n_found}


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    u = len(ta | tb)
    return len(ta & tb) / u if u else 0.0


def extract_features(
    candidates_flat: pd.DataFrame,
    lookup: pd.DataFrame,
    gt_lookup: dict[str, frozenset] | None,
    out_path: pathlib.Path,
    chunk: int = 200_000,
) -> None:
    """Extract features for candidate pairs, writing to TSV."""
    log.info("  Feature extraction → %s", out_path)
    n_total = len(candidates_flat)
    n_chunks = max(1, (n_total + chunk - 1) // chunk)
    first = True

    for ci in range(n_chunks):
        slc = candidates_flat.iloc[ci * chunk: (ci + 1) * chunk].reset_index(drop=True)
        valid = slc["s1_id"].isin(lookup.index) & slc["target_id"].isin(lookup.index)
        slc = slc[valid].reset_index(drop=True)
        if slc.empty:
            continue

        s1v  = lookup.reindex(slc["s1_id"].values)
        tgtv = lookup.reindex(slc["target_id"].values)
        s1v.index  = slc.index
        tgtv.index = slc.index

        feat = pd.DataFrame(index=slc.index)
        feat["s1_id"]     = slc["s1_id"].values
        feat["target_id"] = slc["target_id"].values

        # RapidFuzz batch
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

        feat["name_sorted_exact"]   = (s1v["name_sorted"].values   == tgtv["name_sorted"].values).astype("float32")
        feat["name_alphanum_exact"] = (s1v["name_alphanum"].values  == tgtv["name_alphanum"].values).astype("float32")

        feat["addr_token_jaccard"] = [_jaccard(a, b)
                                      for a, b in zip(s1v["addr_alphanum"], tgtv["addr_alphanum"])]
        feat["addr_numeric_match"] = (
            (s1v["addr_numeric"].values == tgtv["addr_numeric"].values) &
            (s1v["addr_numeric"].str.len() >= 2).values
        ).astype("float32")

        feat["country_match"]        = (s1v["country"].values == tgtv["country"].values).astype("float32")

        a_len = np.maximum(s1v["name_alphanum"].str.len().values, 1)
        b_len = np.maximum(tgtv["name_alphanum"].str.len().values, 1)
        feat["name_len_ratio"]       = (np.minimum(a_len, b_len) / np.maximum(a_len, b_len)).astype("float32")

        a_tok = np.maximum(s1v["name_alphanum"].str.split().str.len().values, 1)
        b_tok = np.maximum(tgtv["name_alphanum"].str.split().str.len().values, 1)
        feat["name_token_len_ratio"] = (np.minimum(a_tok, b_tok) / np.maximum(a_tok, b_tok)).astype("float32")

        if gt_lookup is not None:
            feat["label"] = [1 if slc.loc[i, "target_id"] in gt_lookup.get(slc.loc[i, "s1_id"], frozenset()) else 0
                             for i in slc.index]

        feat.to_csv(out_path, sep="\t", index=False,
                    header=first, mode="w" if first else "a", encoding="utf-8")
        first = False
        log.info("    chunk %d/%d  %d rows", ci + 1, n_chunks, len(feat))

    log.info("  Done: %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — LightGBM train + threshold
# ═══════════════════════════════════════════════════════════════════════════

def _f05(probs: np.ndarray, labels: np.ndarray, t: float) -> float:
    preds  = (probs >= t).astype(int)
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

    model = lgb.LGBMClassifier(
        objective="binary", boosting_type="gbdt", num_leaves=63,
        learning_rate=0.05, n_estimators=500, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=spw, random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_out))
    log.info("  Model saved: %s", model_out)

    # Feature importance
    for name, imp in sorted(zip(FEAT_COLS, model.feature_importances_), key=lambda x: -x[1]):
        log.info("    %-35s %d", name, imp)

    # Threshold tuning
    va_probs = model.predict_proba(X_va)[:, 1]
    best_t   = max(np.arange(0.20, 0.91, 0.01), key=lambda t: _f05(va_probs, y_va, t))
    best_f   = _f05(va_probs, y_va, best_t)
    log.info("  Best threshold: %.3f  →  F0.5=%.4f", best_t, best_f)
    threshold_out.write_text(f"{best_t:.4f}\n", encoding="utf-8")
    return best_t


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Calibration
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Prediction helper
# ═══════════════════════════════════════════════════════════════════════════

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
    for chunk_df in pd.read_csv(feat_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk):
        for c in FEAT_COLS:
            if c in chunk_df.columns:
                chunk_df[c] = pd.to_numeric(chunk_df[c], errors="coerce").fillna(0)
        X    = chunk_df[FEAT_COLS].values.astype("float32")
        raw  = booster.predict(X)
        cal  = iso.predict(raw).astype("float32")
        out_chunk = chunk_df[["s1_id", "target_id"]].copy()
        out_chunk["cal_prob"] = cal
        out_chunk.to_csv(out_path, sep="\t", index=False,
                         header=first, mode="w" if first else "a", encoding="utf-8")
        first = False
    log.info("  Scores written: %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6 — Entity-level decision
# ═══════════════════════════════════════════════════════════════════════════

def resolve(
    scores_path: pathlib.Path,
    all_s1_ids: set[str],
    threshold: float,
    output_path: pathlib.Path,
    chunk: int = 1_000_000,
) -> None:
    log.info("=== PHASE 8: Decision layer (threshold=%.4f) ===", threshold)

    chunks = []
    for df in pd.read_csv(scores_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk):
        df["cal_prob"] = pd.to_numeric(df["cal_prob"], errors="coerce").fillna(0.0)
        above = df[df["cal_prob"] >= threshold][["s1_id", "target_id", "cal_prob"]].copy()
        if not above.empty:
            chunks.append(above)

    matched = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
        columns=["s1_id", "target_id", "cal_prob"])
    log.info("  Pairs above threshold: %d", len(matched))

    # Target-side dedup (greedy, highest score wins)
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
    assert len(out_df) == len(all_s1_ids), "ID set mismatch!"
    log.info("  Gate PASSED: all %d S1 IDs present in output.", len(all_s1_ids))


# ═══════════════════════════════════════════════════════════════════════════
# TEST BLOCKING  (DuckDB, same 5 channels)
# ═══════════════════════════════════════════════════════════════════════════

def block_test(
    s1n: pd.DataFrame,
    s2s3n: pd.DataFrame,
    flat_out: pathlib.Path,
    grouped_out: pathlib.Path,
) -> None:
    log.info("=== PHASE 15a: Test Blocking ===")
    t0 = time.time()
    con = duckdb.connect(database=":memory:", config={"threads": os.cpu_count() or 14})
    con.register("s1n",   s1n)
    con.register("s2s3n", s2s3n)

    all_p = []

    ch1 = con.execute("""
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'exact_sorted' AS channel
        FROM s1n a JOIN s2s3n b ON a.name_sorted = b.name_sorted
        WHERE length(a.name_sorted) >= 3 AND a.entity_id <> b.entity_id""").df()
    all_p.append(ch1); log.info("  CH1: %d", len(ch1))

    ch2 = con.execute("""
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'exact_expanded_sorted' AS channel
        FROM s1n a JOIN s2s3n b ON a.name_expanded_sorted = b.name_expanded_sorted
        WHERE length(a.name_expanded_sorted) >= 3 AND a.entity_id <> b.entity_id""").df()
    all_p.append(ch2); log.info("  CH2: %d", len(ch2))

    s23_tok = (s2s3n[["entity_id","name_alphanum"]].assign(
        token=s2s3n["name_alphanum"].str.split()).explode("token"))
    s23_tok = s23_tok[s23_tok["token"].str.len() >= MIN_TOK_LEN].copy()
    freq = s23_tok.groupby("token")["entity_id"].nunique()
    tok_rare   = s23_tok[s23_tok["token"].isin(freq[freq <= MAX_RARE_FREQ].index)][["token","entity_id"]].drop_duplicates()
    tok_medium = s23_tok[s23_tok["token"].isin(freq[freq <= MAX_MED_FREQ].index)][["token","entity_id"]].drop_duplicates()
    s1_tok = (s1n[["entity_id","name_alphanum"]].assign(
        token=s1n["name_alphanum"].str.split()).explode("token"))
    s1_tok = s1_tok[s1_tok["token"].str.len() >= MIN_TOK_LEN].copy()

    con.register("s1_tok",    s1_tok)
    con.register("tok_rare",  tok_rare)
    con.register("tok_medium", tok_medium)

    ch3 = con.execute("""
        SELECT DISTINCT s.entity_id AS s1_id, t.entity_id AS target_id, 'token_rare' AS channel
        FROM s1_tok s JOIN tok_rare t ON s.token = t.token WHERE s.entity_id <> t.entity_id""").df()
    all_p.append(ch3); log.info("  CH3: %d", len(ch3))

    ch4 = con.execute("""
        SELECT s.entity_id AS s1_id, t.entity_id AS target_id, 'token_medium' AS channel
        FROM s1_tok s JOIN tok_medium t ON s.token = t.token WHERE s.entity_id <> t.entity_id
        GROUP BY s.entity_id, t.entity_id HAVING COUNT(*) >= 2""").df()
    all_p.append(ch4); log.info("  CH4: %d", len(ch4))

    # addr key
    def _addr_key2(s: pd.Series) -> pd.Series:
        ex = s.str.extract(r"(?<!\d)(\d{4,})\s+([a-z]{4,})", expand=True)
        return (ex[0].fillna("") + " " + ex[1].fillna("")).str.strip()

    for df_ref in [s1n, s2s3n]:
        df_ref["addr_key"] = _addr_key2(df_ref["addr_alphanum"])
    con.register("s1n",   s1n)
    con.register("s2s3n", s2s3n)
    ch5 = con.execute("""
        SELECT a.entity_id AS s1_id, b.entity_id AS target_id, 'addr_composite' AS channel
        FROM s1n a JOIN s2s3n b ON a.addr_key = b.addr_key AND a.country = b.country
        WHERE length(a.addr_key) >= 8 AND a.entity_id <> b.entity_id""").df()
    all_p.append(ch5); log.info("  CH5: %d", len(ch5))

    union = pd.concat(all_p, ignore_index=True).drop_duplicates(subset=["s1_id","target_id"], keep="first")
    log.info("  Total test candidates: %d  (%.1f min)", len(union), (time.time()-t0)/60)

    # Flat for features
    flat_out.parent.mkdir(parents=True, exist_ok=True)
    union.rename(columns={"s1_id":"source1_entity_id","target_id":"candidate_entity_id"}).to_csv(
        flat_out, sep="\t", index=False, encoding="utf-8")

    # Grouped for validator
    grouped = (union.groupby("s1_id")["target_id"]
               .apply(lambda x: ",".join(x.tolist()))
               .reset_index()
               .rename(columns={"s1_id":"source1_entity_id","target_id":"candidate_entity_ids"}))
    all_s1 = pd.DataFrame({"source1_entity_id": s1n["entity_id"].tolist()})
    grouped = all_s1.merge(grouped, on="source1_entity_id", how="left")
    grouped["candidate_entity_ids"] = grouped["candidate_entity_ids"].fillna("")
    grouped_out.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(grouped_out, sep="\t", index=False, encoding="utf-8")
    log.info("  candidate_pairs.tsv written: %s (%d rows)", grouped_out, len(grouped))
    con.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    T_START = time.time()
    log.info("=" * 70)
    log.info("MASTER RUN — Option A Fast-Track Pipeline")
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
    s1  = pd.read_csv(TRAIN_DIR / "train_source1.tsv",  sep="\t", dtype=str, keep_default_na=False)
    s2  = pd.read_csv(TRAIN_DIR / "train_source2.tsv",  sep="\t", dtype=str, keep_default_na=False)
    s3  = pd.read_csv(TRAIN_DIR / "train_source3.tsv",  sep="\t", dtype=str, keep_default_na=False)
    gt  = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    log.info("  S1=%d  S2=%d  S3=%d", len(s1), len(s2), len(s3))

    s1n   = _norm(s1)
    s2n   = _norm(s2)
    s3n   = _norm(s3)
    s2s3n = pd.concat([s2n, s3n], ignore_index=True)

    # Build GT lookup
    gt_lookup: dict[str, frozenset] = {}
    for _, row in gt.iterrows():
        raw = row["matched_entity_ids"].strip()
        gt_lookup[row["source1_entity_id"]] = frozenset(raw.split(",")) if raw else frozenset()

    # ── Phase 4: Blocking (all folds) ──────────────────────────────────────
    flat_train = ARTS / "candidate_pairs_all_folds_flat.tsv"
    stats4 = phase4_blocking(s1n, s2s3n, gt, split_df, flat_train)
    if stats4["gate"] != "PASS":
        log.warning("Phase 4 gate not met (recall=%.4f < 0.95) — continuing anyway.", stats4["recall"])

    # Build normalized lookup for features
    lookup = pd.concat([s1n, s2s3n], ignore_index=True).set_index("entity_id")

    # ── Phase 5: Feature extraction for folds A, B, C ──────────────────────
    log.info("\n=== PHASE 5: Feature Extraction ===")
    cands_all = pd.read_csv(flat_train, sep="\t", dtype=str, keep_default_na=False)
    # rename to consistent col names
    if "source1_entity_id" in cands_all.columns:
        cands_all = cands_all.rename(columns={"source1_entity_id":"s1_id","candidate_entity_id":"target_id"})

    for fold_name in ("A", "B", "C"):
        fold_ids = folds[fold_name]
        fold_cands = cands_all[cands_all["s1_id"].isin(fold_ids)].reset_index(drop=True)
        log.info("  Fold %s: %d candidate pairs", fold_name, len(fold_cands))
        feat_path = ARTS / f"features_fold_{fold_name.lower()}.tsv"
        if feat_path.exists():
            log.info("  (already exists, skipping)")
            continue
        extract_features(fold_cands, lookup, gt_lookup, feat_path)

    # ── Phase 6: LightGBM ──────────────────────────────────────────────────
    model_path     = ARTS / "lgbm_model.txt"
    threshold_path = ARTS / "threshold.txt"
    best_t = train_model(
        ARTS / "features_fold_a.tsv",
        ARTS / "features_fold_b.tsv",
        model_path,
        threshold_path,
    )

    # ── Phase 7: Calibration ───────────────────────────────────────────────
    cal_path = ARTS / "calibrator.pkl"
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

    test_flat_cands  = ARTS / "test_candidates_flat.tsv"
    test_cands_tsv   = OUT_DIR / "candidate_pairs.tsv"
    block_test(ts1n, ts2s3n, test_flat_cands, test_cands_tsv)

    # Feature extraction for test
    test_lookup = pd.concat([ts1n, ts2s3n], ignore_index=True).set_index("entity_id")
    test_cands_df = pd.read_csv(test_flat_cands, sep="\t", dtype=str, keep_default_na=False)
    test_cands_df = test_cands_df.rename(columns={"source1_entity_id":"s1_id","candidate_entity_id":"target_id"})
    feat_test = ARTS / "features_test.tsv"
    extract_features(test_cands_df, test_lookup, None, feat_test)

    # Predict + calibrate
    cal_scores_test = ARTS / "cal_scores_test.tsv"
    predict_and_calibrate(feat_test, model_path, cal_path, cal_scores_test)

    # Decision
    all_test_s1 = set(ts1["entity_id"].tolist())
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
