"""
matcher.py -- Phase 6: LightGBM pairwise matching model.

Trains a binary LightGBM classifier on feature matrices from Phase 5.
Uses hard-negative mining: negatives are blocker candidates that are
NOT true matches (as opposed to random entity pairs).

Training data: fold A features (artifacts/features_fold_a.tsv)
Validation  : fold B features (artifacts/features_fold_b.tsv)
Model saved : artifacts/lgbm_model.txt

Usage:
  python src/models/matcher.py --train
  python src/models/matcher.py --predict --features artifacts/features_test.tsv \\
                                          --output  artifacts/scores_test.tsv
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

_REPO = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO))
from config.config import load_config

logger = logging.getLogger(__name__)

# Feature columns used by the model (must match pairwise.py output)
FEATURE_COLS = [
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_expanded_sort_ratio",
    "name_sorted_exact",
    "name_alphanum_exact",
    "addr_token_jaccard",
    "addr_numeric_match",
    "country_match",
    "name_len_ratio",
    "name_token_len_ratio",
]

LGBM_PARAMS: dict[str, Any] = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "boosting_type":    "gbdt",
    "num_leaves":       63,
    "learning_rate":    0.05,
    "n_estimators":     500,
    "min_child_samples": 30,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "random_state":     42,
    "n_jobs":           -1,          # use all CPU threads
    "verbose":          -1,
}


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------

def _load_features(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    logger.info("Loaded features %s: %d rows", path.name, len(df))
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Grid-search threshold that maximises macro F0.5 on the given set.
    F0.5 weights precision 2x over recall.
    """
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.20, 0.90, 0.01):
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta_sq   = 0.25
        denom     = beta_sq * precision + recall
        f05       = (1 + beta_sq) * precision * recall / denom if denom > 0 else 0.0
        if f05 > best_f:
            best_f, best_t = f05, t
    logger.info("Best threshold on fold B: %.3f  →  F0.5 = %.4f", best_t, best_f)
    return best_t


def train(
    fold_a_path: pathlib.Path,
    fold_b_path: pathlib.Path,
    model_out: pathlib.Path,
    threshold_out: pathlib.Path,
) -> None:
    if not _HAS_LGB:
        raise ImportError("lightgbm is not installed: pip install lightgbm")

    logger.info("=== Phase 6: LightGBM Training ===")

    train_df = _load_features(fold_a_path)
    val_df   = _load_features(fold_b_path)

    X_train = train_df[FEATURE_COLS].values.astype("float32")
    y_train = train_df["label"].astype(int).values
    X_val   = val_df[FEATURE_COLS].values.astype("float32")
    y_val   = val_df["label"].astype(int).values

    pos_train = int(y_train.sum())
    neg_train = int((y_train == 0).sum())
    logger.info("Train: %d pos, %d neg  (ratio %.2f:1)",
                pos_train, neg_train, neg_train / max(pos_train, 1))

    # Adjust scale_pos_weight for class imbalance
    spw = max(1, neg_train // max(pos_train, 1))
    params = {**LGBM_PARAMS, "scale_pos_weight": spw}

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False),
                   lgb.log_evaluation(50)],
    )

    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_out))
    logger.info("Model saved: %s", model_out)

    # Feature importance log
    importances = sorted(zip(FEATURE_COLS, model.feature_importances_),
                         key=lambda x: -x[1])
    logger.info("Feature importances:")
    for name, imp in importances:
        logger.info("  %-35s %d", name, imp)

    # Threshold tuning on fold B
    val_probs = model.predict_proba(X_val)[:, 1]
    best_t = find_best_threshold(val_probs, y_val)

    threshold_out.parent.mkdir(parents=True, exist_ok=True)
    threshold_out.write_text(f"{best_t:.4f}\n", encoding="utf-8")
    logger.info("Threshold saved: %s (t=%.4f)", threshold_out, best_t)


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------

def predict(
    features_path: pathlib.Path,
    model_path: pathlib.Path,
    output_path: pathlib.Path,
    chunk_size: int = 500_000,
) -> None:
    if not _HAS_LGB:
        raise ImportError("lightgbm is not installed")

    logger.info("=== Phase 6: Prediction ===")
    booster = lgb.Booster(model_file=str(model_path))
    logger.info("Model loaded: %s", model_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True
    n_written = 0

    for chunk in pd.read_csv(
        features_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size
    ):
        for col in FEATURE_COLS:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0.0)

        X = chunk[FEATURE_COLS].values.astype("float32")
        probs = booster.predict(X)

        out = chunk[["s1_id", "target_id"]].copy()
        out["score"] = probs.astype("float32")

        out.to_csv(
            output_path, sep="\t", index=False,
            header=first_chunk, mode="w" if first_chunk else "a",
            encoding="utf-8",
        )
        first_chunk = False
        n_written += len(out)
        logger.info("  Predicted chunk: %d rows (total %d)", len(out), n_written)

    logger.info("Scores written: %s (%d rows)", output_path, n_written)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6: LightGBM matcher")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train",   action="store_true")
    mode.add_argument("--predict", action="store_true")
    p.add_argument("--fold-a",    type=pathlib.Path, default=None)
    p.add_argument("--fold-b",    type=pathlib.Path, default=None)
    p.add_argument("--features",  type=pathlib.Path, default=None)
    p.add_argument("--model",     type=pathlib.Path, default=None)
    p.add_argument("--threshold-file", type=pathlib.Path, default=None)
    p.add_argument("--output",    type=pathlib.Path, default=None)
    p.add_argument("--chunk",     type=int, default=500_000)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    cfg  = load_config()
    arts = pathlib.Path(cfg["paths"]["artifacts_dir"])
    args = _parse_args()

    _model     = args.model or arts / "lgbm_model.txt"
    _threshold = args.threshold_file or arts / "threshold.txt"

    if args.train:
        _fold_a = args.fold_a or arts / "features_fold_a.tsv"
        _fold_b = args.fold_b or arts / "features_fold_b.tsv"
        train(_fold_a, _fold_b, _model, _threshold)
    else:
        if not args.features or not args.output:
            raise SystemExit("--predict requires --features and --output")
        predict(args.features, _model, args.output, chunk_size=args.chunk)
