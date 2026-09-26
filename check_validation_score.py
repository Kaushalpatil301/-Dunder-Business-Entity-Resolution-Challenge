#!/usr/bin/env python
"""
check_validation_score.py — Standalone validation scoring tool.

Evaluates the trained LightGBM model and Isotonic Calibrator against
Fold B validation entities using the official competition Macro F0.5 metric.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "code" / "business_entity_resolution" / "src"))
from evaluation.scorer import macro_f05

# Correct project paths
MODEL_FILE         = REPO / "artifacts" / "lgbm_model.txt"
CALIBRATOR_FILE    = REPO / "artifacts" / "calibrator.pkl"
THRESHOLD_FILE     = REPO / "artifacts" / "threshold.txt"
VALIDATION_PARQUET = REPO / "artifacts" / "features_fold_b.parquet"
GROUND_TRUTH_FILE  = REPO / "dataset" / "train" / "train_ground_truth.tsv"
SPLIT_FILE         = REPO / "artifacts" / "entity_split.tsv"

FEAT_COLS = [
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_expanded_sort_ratio",
    "name_qratio",
    "name_jaro_winkler",
    "name_sorted_exact",
    "name_alphanum_exact",
    "name_nospace_exact",
    "addr_digits_exact",
    "addr_token_jaccard",
    "addr_numeric_match",
    "addr_jaro_winkler",
    "addr_prefix4_match",
    "country_match",
    "name_len_ratio",
    "name_token_len_ratio",
    "name_prefix5_exact",
]


def load_ground_truth(validation_entity_ids: set[str]) -> dict[str, set[str]]:
    print(f"Loading ground truth for {len(validation_entity_ids):,} validation entities...")
    df = pd.read_csv(GROUND_TRUTH_FILE, sep="\t", dtype=str, keep_default_na=False)
    truth = {}
    for _, row in df.iterrows():
        s1 = row["source1_entity_id"]
        if s1 not in validation_entity_ids:
            continue
        val = row["matched_entity_ids"].strip()
        truth[s1] = set(val.split(",")) if val else set()
    return truth


def main():
    print("=" * 70)
    print("LIGHTGBM VALIDATION & MACRO F0.5 SCORER")
    print("=" * 70)

    # 1. Load model and calibrator
    print("\n[1] Loading trained model and calibrator...")
    booster = lgb.Booster(model_file=str(MODEL_FILE))
    with open(CALIBRATOR_FILE, "rb") as f:
        iso = pickle.load(f)

    opt_thresh = float(THRESHOLD_FILE.read_text().strip()) if THRESHOLD_FILE.exists() else 0.63
    print(f"  Model loaded: {MODEL_FILE.name}")
    print(f"  Calibrator loaded: {CALIBRATOR_FILE.name}")
    print(f"  Pipeline tuned optimal threshold: {opt_thresh:.4f}")

    # 2. Load validation features
    print("\n[2] Loading validation features from Parquet...")
    df = pd.read_parquet(VALIDATION_PARQUET)
    print(f"  Validation pairs loaded: {len(df):,}")

    # 3. Predict & Calibrate
    print("\n[3] Predicting & Calibrating probabilities...")
    X = df[FEAT_COLS].values.astype(np.float32)
    raw_probs = booster.predict(X, num_threads=8)
    cal_probs = iso.predict(raw_probs)
    df["cal_probability"] = cal_probs

    print(f"  Raw prob min/mean/max : {raw_probs.min():.4f} / {raw_probs.mean():.4f} / {raw_probs.max():.4f}")
    print(f"  Cal prob min/mean/max : {cal_probs.min():.4f} / {cal_probs.mean():.4f} / {cal_probs.max():.4f}")

    # 4. Load Fold B S1 IDs & Ground Truth
    print("\n[4] Loading Fold B entity split & ground truth...")
    split_df = pd.read_csv(SPLIT_FILE, sep="\t", dtype=str)
    fold_b_s1_ids = set(split_df[split_df["fold"] == "B"]["source1_entity_id"])
    print(f"  Total Fold B S1 entities: {len(fold_b_s1_ids):,}")

    truth = load_ground_truth(fold_b_s1_ids)

    # 5. Threshold sweep using official macro_f05
    print("\n" + "=" * 70)
    print("SWEEPING CALIBRATED THRESHOLDS USING OFFICIAL MACRO F0.5")
    print("=" * 70)
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.63, 0.65, 0.70, 0.75, 0.80, 0.85]
    results = []

    for t in thresholds:
        above = df[df["cal_probability"] >= t].sort_values("cal_probability", ascending=False)
        # Greedy target deduplication
        above = above.drop_duplicates(subset=["target_id"], keep="first")
        
        preds = {s: set() for s in fold_b_s1_ids}
        for s, grp in above.groupby("s1_id"):
            if s in preds:
                preds[s] = set(grp["target_id"].tolist()[:15])

        score = macro_f05(truth, preds)
        links = sum(len(m) for m in preds.values())
        results.append((t, score, links))
        print(f"  Threshold={t:.2f}  |  Macro F0.5 = {score:.6f}  |  Predicted links = {links:,}")

    best = max(results, key=lambda x: x[1])
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Best Threshold   : {best[0]:.2f}")
    print(f"  Best Macro F0.5  : {best[1]:.6f}")
    print(f"  Predicted Links  : {best[2]:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
