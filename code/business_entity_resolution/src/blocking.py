"""
blocking.py -- Entry-point script for candidate-pair generation on any split.

Wraps the blocking union (src/blocking/union.py) and feature normalizer to
produce a valid candidate_pairs.tsv for both training and test sets.

Usage (test set — what the reproduction README calls):
  python src/blocking.py \\
      --input_dir ../../dataset/test \\
      --output    ../../output/candidate_pairs.tsv

Usage (train set + recall measurement):
  python src/blocking.py \\
      --input_dir ../../dataset/train \\
      --output    ../../artifacts/candidate_pairs_v1.tsv \\
      --measure-recall
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import pandas as pd

_SRC  = pathlib.Path(__file__).resolve().parent
_REPO = _SRC.parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

from blocking.fast_normalizer import normalize_for_blocking
from blocking.union import (
    exact_key_channel, token_channel, address_channel, build_token_table,
    explode_gt, measure_recall,
    MAX_FREQ_RARE, MAX_FREQ_MEDIUM, MIN_SHARED_RARE, MIN_SHARED_MEDIUM,
)
from config.config import load_config

logger = logging.getLogger(__name__)


def _load_tsv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    logger.info("Loaded %s (%d rows)", path.name, len(df))
    return df


def _detect_prefix(input_dir: pathlib.Path) -> str:
    if (input_dir / "train_source1.tsv").exists():
        return "train"
    if (input_dir / "test_source1.tsv").exists():
        return "test"
    raise FileNotFoundError(f"No source1.tsv found in {input_dir}")


def generate_candidates(
    input_dir: pathlib.Path,
    output_path: pathlib.Path,
    measure_recall_flag: bool = False,
    gt_path: pathlib.Path | None = None,
    fold_d_ids: set[str] | None = None,
) -> None:
    prefix = _detect_prefix(input_dir)

    s1  = _load_tsv(input_dir / f"{prefix}_source1.tsv")
    s2  = _load_tsv(input_dir / f"{prefix}_source2.tsv")
    s3  = _load_tsv(input_dir / f"{prefix}_source3.tsv")

    logger.info("Normalizing ...")
    s1n   = normalize_for_blocking(s1)
    s2n   = normalize_for_blocking(s2)
    s3n   = normalize_for_blocking(s3)
    s2s3n = pd.concat([s2n, s3n], ignore_index=True)

    tok_rare   = build_token_table(s2s3n, MAX_FREQ_RARE,   "rare")
    tok_medium = build_token_table(s2s3n, MAX_FREQ_MEDIUM, "medium")

    all_cands = []
    all_cands.append(exact_key_channel(s1n, s2s3n, "name_sorted",          "exact_sorted"))
    all_cands.append(exact_key_channel(s1n, s2s3n, "name_expanded_sorted", "exact_expanded_sorted"))
    all_cands.append(token_channel(s1n, tok_rare,   MIN_SHARED_RARE,   "token_rare"))
    all_cands.append(token_channel(s1n, tok_medium, MIN_SHARED_MEDIUM, "token_medium"))
    all_cands.append(address_channel(s1n, s2s3n))

    union_df = pd.concat(all_cands, ignore_index=True)
    union_df = union_df.drop_duplicates(subset=["s1_id", "target_id"], keep="first")
    logger.info("Total candidates: %d", len(union_df))

    if measure_recall_flag and gt_path is not None and fold_d_ids is not None:
        gt  = _load_tsv(gt_path)
        tp  = explode_gt(gt)
        stats = measure_recall(union_df, tp, fold_d_ids)
        logger.info("Recall report: %s", stats)

    # Write grouped format: source1_entity_id | candidate_entity_ids (comma-sep)
    grouped = (
        union_df
        .groupby("s1_id")["target_id"]
        .apply(lambda x: ",".join(x.tolist()))
        .reset_index()
        .rename(columns={"s1_id": "source1_entity_id", "target_id": "candidate_entity_ids"})
    )
    all_s1_df = pd.DataFrame({"source1_entity_id": s1["entity_id"].tolist()})
    grouped   = all_s1_df.merge(grouped, on="source1_entity_id", how="left")
    grouped["candidate_entity_ids"] = grouped["candidate_entity_ids"].fillna("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    logger.info("Written: %s (%d rows)", output_path, len(grouped))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blocking entry-point")
    p.add_argument("--input_dir",      required=True, type=pathlib.Path)
    p.add_argument("--output",         required=True, type=pathlib.Path)
    p.add_argument("--measure-recall", action="store_true")
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
    cfg  = load_config()

    generate_candidates(
        input_dir=args.input_dir,
        output_path=args.output,
        measure_recall_flag=args.measure_recall,
    )
