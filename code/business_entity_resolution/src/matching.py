"""
matching.py -- Entry-point script for full matching (feature + model + decision).

Wraps Phases 5-8: takes candidate pairs + source TSVs, runs the trained
LightGBM model through calibration, then writes the final matching_results.tsv.

Usage (as called in the reproduction README):
  python src/matching.py \\
      --candidates ../../output/candidate_pairs.tsv \\
      --input_dir  ../../dataset/test \\
      --output     ../../output/matching_results.tsv

The candidate file must already exist (produced by blocking.py).
The model, calibrator, and threshold must exist under artifacts/.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import pickle
import sys

import pandas as pd

_SRC  = pathlib.Path(__file__).resolve().parent
_REPO = _SRC.parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

from config.config import load_config
from features.pairwise import run as run_features
from models.matcher import predict as predict_scores
from calibration.calibrator import apply_calibration
from decision.resolver import resolve, load_threshold

logger = logging.getLogger(__name__)


def _detect_prefix(input_dir: pathlib.Path) -> str:
    if (input_dir / "train_source1.tsv").exists():
        return "train"
    return "test"


def run_matching(
    candidates_path: pathlib.Path,
    input_dir: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    cfg  = load_config()
    arts = pathlib.Path(cfg["paths"]["artifacts_dir"])

    prefix = _detect_prefix(input_dir)
    s1_path = input_dir / f"{prefix}_source1.tsv"
    s2_path = input_dir / f"{prefix}_source2.tsv"
    s3_path = input_dir / f"{prefix}_source3.tsv"

    # ---- Convert grouped candidate file to flat pairs for feature extraction ----
    logger.info("Loading candidate_pairs.tsv ...")
    cands = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False)

    if "candidate_entity_ids" in cands.columns:
        # grouped format → explode to flat pairs
        flat_rows = []
        for _, row in cands.iterrows():
            s1_id = row["source1_entity_id"]
            raw   = row["candidate_entity_ids"].strip()
            if raw:
                for tgt in raw.split(","):
                    flat_rows.append({"source1_entity_id": s1_id,
                                      "candidate_entity_id": tgt.strip()})
        flat_df = pd.DataFrame(flat_rows)
    else:
        flat_df = cands  # already flat

    flat_cands_path = arts / "_tmp_candidates_flat.tsv"
    flat_df.to_csv(flat_cands_path, sep="\t", index=False, encoding="utf-8")
    logger.info("Flat candidates: %d pairs", len(flat_df))

    # ---- Feature extraction ----
    features_path = arts / "_tmp_features.tsv"
    logger.info("Extracting features ...")
    run_features(
        candidates_path=flat_cands_path,
        output_path=features_path,
        source1_path=s1_path,
        source2_path=s2_path,
        source3_path=s3_path,
    )

    # ---- LightGBM prediction ----
    scores_path = arts / "_tmp_scores.tsv"
    logger.info("Running LightGBM prediction ...")
    predict_scores(
        features_path=features_path,
        model_path=arts / "lgbm_model.txt",
        output_path=scores_path,
    )

    # ---- Calibration ----
    cal_path = arts / "_tmp_cal_scores.tsv"
    logger.info("Applying calibration ...")
    apply_calibration(
        scores_path=scores_path,
        calibrator_path=arts / "calibrator.pkl",
        output_path=cal_path,
    )

    # ---- Decision ----
    logger.info("Entity-level decision ...")
    t = load_threshold(arts / "threshold.txt")
    resolve(
        scores_path=cal_path,
        s1_ids_path=s1_path,
        threshold=t,
        output_path=output_path,
        score_col="cal_prob",
    )

    # ---- Phase 15 gate ----
    pred_ids = set(pd.read_csv(output_path, sep="\t", dtype=str)["source1_entity_id"])
    test_ids = set(pd.read_csv(s1_path,     sep="\t", dtype=str)["entity_id"])
    assert pred_ids == test_ids, (
        f"ID set mismatch: {len(pred_ids - test_ids)} extra, {len(test_ids - pred_ids)} missing"
    )
    logger.info("GATE PASSED: all %d S1 entities present in output.", len(test_ids))
    logger.info("matching_results.tsv: %s", output_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matching entry-point (Phases 5-8)")
    p.add_argument("--candidates", required=True, type=pathlib.Path)
    p.add_argument("--input_dir",  required=True, type=pathlib.Path)
    p.add_argument("--output",     required=True, type=pathlib.Path)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    args = _parse_args()
    run_matching(
        candidates_path=args.candidates,
        input_dir=args.input_dir,
        output_path=args.output,
    )
