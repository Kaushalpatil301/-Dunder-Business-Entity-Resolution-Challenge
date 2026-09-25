"""
run_inference.py -- Phase 15: Full test-set inference pipeline.

Steps:
  1. Block test S1 vs test S2+S3  → output/candidate_pairs.tsv
  2. Extract pairwise features    → artifacts/features_test.tsv
  3. LightGBM prediction          → artifacts/scores_test.tsv
  4. Isotonic calibration         → artifacts/cal_scores_test.tsv
  5. Entity-level decision        → output/matching_results.tsv
  6. Assert set(pred_ids) == set(test_s1_ids)   [Phase 15 gate]

Usage:
  python src/pipeline/run_inference.py
"""

from __future__ import annotations

import logging
import pathlib
import sys

import pandas as pd

_REPO = pathlib.Path(__file__).resolve().parents[5]
_SRC  = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

from config.config import load_config
from blocking.fast_normalizer import normalize_for_blocking
from blocking.union import (
    exact_key_channel, token_channel, address_channel,
    build_token_table, MAX_FREQ_RARE, MAX_FREQ_MEDIUM,
    MIN_SHARED_RARE, MIN_SHARED_MEDIUM,
)
from features.pairwise import run as run_features
from models.matcher import predict as predict_scores
from calibration.calibrator import apply_calibration
from decision.resolver import resolve, load_threshold

logger = logging.getLogger(__name__)


def _load_tsv(path: pathlib.Path, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, **kw)
    logger.info("Loaded %s: %d rows", path.name, len(df))
    return df


def generate_test_candidates(
    test_dir: pathlib.Path,
    output_path: pathlib.Path,
    arts: pathlib.Path,
) -> None:
    """
    Run the 5-channel blocking union on the test set and write candidate_pairs.tsv
    in the format required by the validator: source1_entity_id | candidate_entity_ids.
    """
    logger.info("=== Phase 15 Step 1: Test Blocking ===")

    s1  = _load_tsv(test_dir / "test_source1.tsv")
    s2  = _load_tsv(test_dir / "test_source2.tsv")
    s3  = _load_tsv(test_dir / "test_source3.tsv")

    logger.info("Normalizing test sources ...")
    s1n    = normalize_for_blocking(s1)
    s2n    = normalize_for_blocking(s2)
    s3n    = normalize_for_blocking(s3)
    s2s3n  = pd.concat([s2n, s3n], ignore_index=True)

    logger.info("Building token tables ...")
    tok_rare   = build_token_table(s2s3n, MAX_FREQ_RARE,   "rare")
    tok_medium = build_token_table(s2s3n, MAX_FREQ_MEDIUM, "medium")

    all_cands = []

    logger.info("CH1: exact_sorted ...")
    ch1 = exact_key_channel(s1n, s2s3n, "name_sorted",          "exact_sorted")
    all_cands.append(ch1)

    logger.info("CH2: exact_expanded_sorted ...")
    ch2 = exact_key_channel(s1n, s2s3n, "name_expanded_sorted", "exact_expanded_sorted")
    all_cands.append(ch2)

    logger.info("CH3: token_rare ...")
    ch3 = token_channel(s1n, tok_rare,   MIN_SHARED_RARE,   "token_rare")
    all_cands.append(ch3)

    logger.info("CH4: token_medium ...")
    ch4 = token_channel(s1n, tok_medium, MIN_SHARED_MEDIUM, "token_medium")
    all_cands.append(ch4)

    logger.info("CH5: addr_composite ...")
    ch5 = address_channel(s1n, s2s3n)
    all_cands.append(ch5)

    union_df = pd.concat(all_cands, ignore_index=True)
    union_df = union_df.drop_duplicates(subset=["s1_id", "target_id"], keep="first")
    logger.info("Total candidates after dedup: %d", len(union_df))

    # ----- Write candidate_pairs.tsv in validator-required format -----
    # The validator expects: source1_entity_id | candidate_entity_ids
    # where candidate_entity_ids is a comma-separated list per S1 entity.
    # We write one row per (s1, target) pair so the format matches the
    # per-pair structure used by the feature extractor, but the validator
    # actually accepts both the per-pair and the grouped formats.
    # To stay consistent with the PS spec (one row per S1 entity),
    # we group and write the grouped format.
    grouped = (
        union_df
        .groupby("s1_id")["target_id"]
        .apply(lambda x: ",".join(x.tolist()))
        .reset_index()
        .rename(columns={"s1_id": "source1_entity_id", "target_id": "candidate_entity_ids"})
    )
    # Also add S1 entities that had zero candidates (they must appear as singletons)
    all_s1 = pd.DataFrame({"source1_entity_id": s1["entity_id"].tolist()})
    grouped = all_s1.merge(grouped, on="source1_entity_id", how="left")
    grouped["candidate_entity_ids"] = grouped["candidate_entity_ids"].fillna("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    logger.info("candidate_pairs.tsv written: %s (%d rows)", output_path, len(grouped))

    # Also save a flat pair file for feature extraction (s1_id, candidate_entity_id)
    flat_out = arts / "candidate_pairs_test_flat.tsv"
    flat = union_df.rename(columns={"s1_id": "source1_entity_id",
                                     "target_id": "candidate_entity_id"})
    flat.to_csv(flat_out, sep="\t", index=False, encoding="utf-8")
    logger.info("Flat candidates saved: %s", flat_out)


def run_inference() -> None:
    cfg      = load_config()
    arts     = pathlib.Path(cfg["paths"]["artifacts_dir"])
    test_dir = pathlib.Path(cfg["paths"]["test_dir"])
    out_dir  = pathlib.Path(cfg["paths"]["output_dir"])
    arts.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cands_tsv        = out_dir  / "candidate_pairs.tsv"
    cands_flat       = arts     / "candidate_pairs_test_flat.tsv"
    features_test    = arts     / "features_test.tsv"
    scores_test      = arts     / "scores_test.tsv"
    cal_scores_test  = arts     / "cal_scores_test.tsv"
    matching_results = out_dir  / "matching_results.tsv"
    model_path       = arts     / "lgbm_model.txt"
    calibrator_path  = arts     / "calibrator.pkl"
    threshold_path   = arts     / "threshold.txt"

    # --- Step 1: Blocking ---
    generate_test_candidates(test_dir, cands_tsv, arts)

    # --- Step 2: Feature extraction ---
    logger.info("\n=== Phase 15 Step 2: Feature Extraction ===")
    run_features(
        candidates_path=cands_flat,
        output_path=features_test,
        source1_path=test_dir / "test_source1.tsv",
        source2_path=test_dir / "test_source2.tsv",
        source3_path=test_dir / "test_source3.tsv",
    )

    # --- Step 3: LightGBM prediction ---
    logger.info("\n=== Phase 15 Step 3: LightGBM Prediction ===")
    predict_scores(
        features_path=features_test,
        model_path=model_path,
        output_path=scores_test,
    )

    # --- Step 4: Calibration ---
    logger.info("\n=== Phase 15 Step 4: Apply Calibration ===")
    apply_calibration(
        scores_path=scores_test,
        calibrator_path=calibrator_path,
        output_path=cal_scores_test,
    )

    # --- Step 5: Entity decision ---
    logger.info("\n=== Phase 15 Step 5: Entity-Level Decision ===")
    t = load_threshold(threshold_path)
    resolve(
        scores_path=cal_scores_test,
        s1_ids_path=test_dir / "test_source1.tsv",
        threshold=t,
        output_path=matching_results,
        score_col="cal_prob",
    )

    # --- Step 6: Phase 15 gate assertion ---
    logger.info("\n=== Phase 15 Gate: set(pred_ids) == set(test_s1_ids) ===")
    pred_ids  = set(pd.read_csv(matching_results,  sep="\t", dtype=str)["source1_entity_id"])
    test_ids  = set(pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str)["entity_id"])
    assert pred_ids == test_ids, (
        f"ID set mismatch: {len(pred_ids - test_ids)} extra, {len(test_ids - pred_ids)} missing"
    )
    logger.info("GATE PASSED: all %d test S1 entities are in output.", len(test_ids))
    logger.info("\nInference complete. Outputs:")
    logger.info("  %s", matching_results)
    logger.info("  %s", cands_tsv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    run_inference()
