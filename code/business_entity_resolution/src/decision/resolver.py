"""
resolver.py -- Phase 8: Entity-level decision layer.

Takes calibrated pairwise scores and produces the final matching_results.tsv.

Decision rules (in order):
1. Apply F0.5-optimised threshold: only predict pairs above threshold.
2. Singleton protection: if no pair passes threshold for a given S1,
   output empty matched_entity_ids (correct abstain scores 1.0 for singletons).
3. Target-side deduplication: since target uniqueness holds (Phase 1 EDA),
   if multiple S1 entities claim the same S2/S3 target, assign to the one
   with the highest calibrated probability (greedy by score, desc).
4. Output: every test S1 entity must appear exactly once.

Usage:
  python src/decision/resolver.py \\
      --scores       artifacts/cal_scores_test.tsv \\
      --s1-ids       dataset/test/test_source1.tsv \\
      --threshold    artifacts/threshold.txt \\
      --output       output/matching_results.tsv
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd

_REPO = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO))
from config.config import load_config

logger = logging.getLogger(__name__)


def load_threshold(path: pathlib.Path, default: float = 0.5) -> float:
    if not path.exists():
        logger.warning("Threshold file not found (%s), using default %.3f", path, default)
        return default
    t = float(path.read_text(encoding="utf-8").strip())
    logger.info("Threshold loaded: %.4f", t)
    return t


def resolve(
    scores_path: pathlib.Path,
    s1_ids_path: pathlib.Path,
    threshold: float,
    output_path: pathlib.Path,
    score_col: str = "cal_prob",
    chunk_size: int = 1_000_000,
) -> None:
    """
    scores_path : TSV with columns s1_id, target_id, score[, cal_prob]
    s1_ids_path : test_source1.tsv (all test S1 entity_ids required in output)
    threshold   : minimum score to call a match
    output_path : matching_results.tsv
    """
    logger.info("=== Phase 8: Entity-Level Decision ===")
    logger.info("Threshold: %.4f  |  Score column: %s", threshold, score_col)

    # --- Load all S1 IDs that must appear in output ---
    s1_ref = pd.read_csv(s1_ids_path, sep="\t", dtype=str, keep_default_na=False,
                         usecols=["entity_id"])
    all_s1_ids: set[str] = set(s1_ref["entity_id"].tolist())
    logger.info("Total S1 entities required in output: %d", len(all_s1_ids))

    # --- Load and filter scores above threshold ---
    logger.info("Loading scores from %s ...", scores_path)
    chunks = []
    for chunk in pd.read_csv(
        scores_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size
    ):
        # Use cal_prob if available; fall back to raw score
        if score_col in chunk.columns:
            chunk["_prob"] = pd.to_numeric(chunk[score_col], errors="coerce").fillna(0.0)
        elif "score" in chunk.columns:
            chunk["_prob"] = pd.to_numeric(chunk["score"], errors="coerce").fillna(0.0)
        else:
            raise ValueError(f"No score column found in {scores_path}")

        above = chunk[chunk["_prob"] >= threshold][["s1_id", "target_id", "_prob"]].copy()
        if not above.empty:
            chunks.append(above)

    if chunks:
        matched = pd.concat(chunks, ignore_index=True)
        logger.info("Pairs above threshold: %d", len(matched))
    else:
        matched = pd.DataFrame(columns=["s1_id", "target_id", "_prob"])
        logger.warning("No pairs passed the threshold — all outputs will be singletons!")

    # --- Target-side deduplication (greedy: highest score wins) ---
    if not matched.empty:
        # Sort by prob descending, deduplicate so each target_id is assigned once
        matched = matched.sort_values("_prob", ascending=False)
        matched = matched.drop_duplicates(subset=["target_id"], keep="first")
        logger.info("After target-side dedup: %d pairs", len(matched))

    # --- Build output dict: s1_id → comma-separated matched IDs ---
    result: dict[str, list[str]] = {s1: [] for s1 in all_s1_ids}
    for _, row in matched.iterrows():
        s1 = row["s1_id"]
        tgt = row["target_id"]
        if s1 in result:
            result[s1].append(tgt)

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for s1_id in sorted(result.keys()):
        matches = result[s1_id]
        rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches),
        })

    out_df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    out_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    logger.info("matching_results.tsv written: %s (%d rows)", output_path, len(out_df))

    # --- Stats ---
    n_singleton = int((out_df["matched_entity_ids"] == "").sum())
    n_matched   = int((out_df["matched_entity_ids"] != "").sum())
    logger.info("Singletons (empty): %d  |  Matched: %d", n_singleton, n_matched)
    assert len(out_df) == len(all_s1_ids), (
        f"Output has {len(out_df)} rows but expected {len(all_s1_ids)}"
    )
    logger.info("Assertion passed: set(pred_ids) == set(test_s1_ids)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 8: Entity-level decision layer")
    p.add_argument("--scores",     required=True, type=pathlib.Path)
    p.add_argument("--s1-ids",     required=True, type=pathlib.Path)
    p.add_argument("--threshold",  type=pathlib.Path, default=None)
    p.add_argument("--threshold-value", type=float, default=None)
    p.add_argument("--output",     required=True, type=pathlib.Path)
    p.add_argument("--score-col",  default="cal_prob")
    p.add_argument("--chunk",      type=int, default=1_000_000)
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

    _threshold_path = args.threshold or arts / "threshold.txt"
    if args.threshold_value is not None:
        t = args.threshold_value
    else:
        t = load_threshold(_threshold_path)

    resolve(
        scores_path=args.scores,
        s1_ids_path=args.s1_ids,
        threshold=t,
        output_path=args.output,
        score_col=args.score_col,
        chunk_size=args.chunk,
    )
