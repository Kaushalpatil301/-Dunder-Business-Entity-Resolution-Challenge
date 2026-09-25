"""
calibrator.py -- Phase 7: Isotonic calibration of LightGBM scores.

Uses fold C features (fold-disjoint from training fold A and tuning fold B)
to fit an isotonic regression that maps raw model scores → well-calibrated
probabilities. Saves the calibrator as a pickle for use in the decision layer.

Usage:
  python src/calibration/calibrator.py \\
      --fold-c-scores artifacts/scores_fold_c.tsv \\
      --fold-c-features artifacts/features_fold_c.tsv \\
      --model artifacts/lgbm_model.txt \\
      --output artifacts/calibrator.pkl
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

_REPO = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO))
from config.config import load_config

logger = logging.getLogger(__name__)

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


def fit_calibrator(
    features_path: pathlib.Path,
    model_path: pathlib.Path,
    calibrator_out: pathlib.Path,
) -> None:
    """Fit isotonic calibration on fold C, save calibrator."""
    if not _HAS_LGB:
        raise ImportError("lightgbm not installed")

    logger.info("=== Phase 7: Isotonic Calibration (fold C) ===")

    df = pd.read_csv(features_path, sep="\t", dtype=str, keep_default_na=False)
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "label" not in df.columns:
        raise ValueError("fold C feature file must contain a 'label' column")

    X      = df[FEATURE_COLS].values.astype("float32")
    labels = df["label"].astype(int).values

    booster = lgb.Booster(model_file=str(model_path))
    raw_scores = booster.predict(X)

    logger.info("Fitting isotonic regression on %d pairs ...", len(labels))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_scores, labels)

    calibrator_out.parent.mkdir(parents=True, exist_ok=True)
    with open(calibrator_out, "wb") as f:
        pickle.dump(iso, f)
    logger.info("Calibrator saved: %s", calibrator_out)

    # --- Print reliability diagnostics ---
    prob_true, prob_pred = calibration_curve(labels, iso.predict(raw_scores), n_bins=10)
    logger.info("Calibration curve (pred → actual):")
    for pt, pp in zip(prob_pred, prob_true):
        bar = "#" * int(pp * 20)
        logger.info("  pred=%.2f  actual=%.2f  |%s", pp, pt, bar)


def apply_calibration(
    scores_path: pathlib.Path,
    calibrator_path: pathlib.Path,
    output_path: pathlib.Path,
    chunk_size: int = 500_000,
) -> None:
    """Apply a saved calibrator to a scores TSV (s1_id, target_id, score)."""
    with open(calibrator_path, "rb") as f:
        iso: IsotonicRegression = pickle.load(f)
    logger.info("Calibrator loaded: %s", calibrator_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True
    n_written   = 0

    for chunk in pd.read_csv(
        scores_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size
    ):
        chunk["score"]  = pd.to_numeric(chunk["score"], errors="coerce").fillna(0.0)
        chunk["cal_prob"] = iso.predict(chunk["score"].values).astype("float32")

        chunk[["s1_id", "target_id", "score", "cal_prob"]].to_csv(
            output_path, sep="\t", index=False,
            header=first_chunk, mode="w" if first_chunk else "a",
            encoding="utf-8",
        )
        first_chunk = False
        n_written  += len(chunk)

    logger.info("Calibrated scores written: %s (%d rows)", output_path, n_written)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 7: Isotonic calibration")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit",   action="store_true")
    mode.add_argument("--apply", action="store_true")
    p.add_argument("--features",     type=pathlib.Path)
    p.add_argument("--scores",       type=pathlib.Path)
    p.add_argument("--model",        type=pathlib.Path)
    p.add_argument("--calibrator",   type=pathlib.Path)
    p.add_argument("--output",       type=pathlib.Path)
    p.add_argument("--chunk",        type=int, default=500_000)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    cfg  = load_config()
    arts = pathlib.Path(cfg["paths"]["artifacts_dir"])
    args = _parse_args()

    _model      = args.model      or arts / "lgbm_model.txt"
    _calibrator = args.calibrator or arts / "calibrator.pkl"

    if args.fit:
        _feats = args.features or arts / "features_fold_c.tsv"
        fit_calibrator(_feats, _model, _calibrator)
    else:
        if not args.scores or not args.output:
            raise SystemExit("--apply requires --scores and --output")
        apply_calibration(args.scores, _calibrator, args.output, chunk_size=args.chunk)
