"""
splitter.py -- Phase 2 entity-level train/val split for the Amazon ML Challenge.

Implements a reproducible, stratified split of S1 entities from
train_ground_truth.tsv into four folds:
  A (60%) -- primary training set
  B (10%) -- country-specificity / feature leakage check (Phase 5)
  C (15%) -- isotonic calibration (Phase 7)
  D (15%) -- threshold-tuning / held-out val (Phase 8+)

Stratification preserves the singleton/non-singleton ratio across all folds
(within a few percent), since singletons are a different scoring regime.

All randomness is sourced from config['random_seed'] -- no hardcoded literals.
Output is written to artifacts/entity_split.parquet (one row per S1 entity,
columns: source1_entity_id, is_singleton, n_matches, fold).

Assumption: raw GT file is never modified; the split artifact is the only output.
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap path so config/ is importable from repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
from config.config import load_config

logger = logging.getLogger(__name__)

TRAIN_GROUND_TRUTH = "train_ground_truth.tsv"
OUTPUT_FILE = "entity_split.parquet"


def _parse_n_matches(val: str) -> int:
    """Return number of matches for a GT row (0 = singleton)."""
    stripped = val.strip()
    if not stripped:
        return 0
    return len(stripped.split(","))


def build_entity_frame(gt_path: pathlib.Path) -> pd.DataFrame:
    """Load GT and build one-row-per-S1 dataframe with match count and singleton flag.

    Assumption: GT must be read with sep=TAB; missing sep collapses rows silently.
    """
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    required = {"source1_entity_id", "matched_entity_ids"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"Ground truth missing columns: {missing}")

    gt["n_matches"] = gt["matched_entity_ids"].apply(_parse_n_matches)
    gt["is_singleton"] = gt["n_matches"] == 0
    logger.info(
        "GT loaded: %d entities, %d singletons (%.2f%%)",
        len(gt),
        gt["is_singleton"].sum(),
        100.0 * gt["is_singleton"].mean(),
    )
    return gt[["source1_entity_id", "n_matches", "is_singleton"]]


def stratified_split(
    entity_df: pd.DataFrame,
    fold_sizes: dict[str, float],
    random_seed: int,
) -> pd.DataFrame:
    """Assign each S1 entity to a fold, stratified by is_singleton.

    Uses sequential sampling within each stratum so proportions are exact
    (no sklearn dependency for this step).

    Args:
        entity_df: DataFrame with source1_entity_id, is_singleton columns.
        fold_sizes: Dict mapping fold name to fraction (must sum to 1.0).
        random_seed: RNG seed for reproducibility.

    Returns:
        entity_df with a new 'fold' column.
    """
    total = sum(fold_sizes.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fold_sizes must sum to 1.0, got {total:.6f}")

    folds = list(fold_sizes.keys())
    df = entity_df.copy()
    df["fold"] = ""

    for stratum in [True, False]:
        mask = df["is_singleton"] == stratum
        stratum_df = df[mask].sample(frac=1, random_state=random_seed)
        n = len(stratum_df)
        idx = stratum_df.index.tolist()

        start = 0
        for i, fold_name in enumerate(folds):
            if i == len(folds) - 1:
                end = n  # last fold gets remainder
            else:
                end = start + round(fold_sizes[fold_name] * n)
            df.loc[idx[start:end], "fold"] = fold_name
            start = end

    return df


def print_split_stats(df: pd.DataFrame) -> None:
    """Print per-fold statistics (count, singleton %, non-singleton %) for gate check."""
    total = len(df)
    print("=" * 65)
    print("PHASE 2 SPLIT STATISTICS")
    print("=" * 65)
    print(f"  Total S1 entities: {total:,}")
    print()
    print(f"  {'Fold':<6} {'N':>9} {'%Total':>8} {'Singletons':>12} {'Sing%':>8} {'Multi%':>8}")
    print("  " + "-" * 55)

    train_sing_pct = None
    for fold in sorted(df["fold"].unique()):
        sub = df[df["fold"] == fold]
        n = len(sub)
        pct_total = 100.0 * n / total
        n_sing = sub["is_singleton"].sum()
        sing_pct = 100.0 * n_sing / n
        multi_pct = 100.0 - sing_pct
        flag = ""
        if fold == "A":
            train_sing_pct = sing_pct
        elif train_sing_pct is not None:
            diff = abs(sing_pct - train_sing_pct)
            flag = f"  (delta vs A: {diff:+.2f}pp)"
        print(f"  {fold:<6} {n:>9,} {pct_total:>8.2f}% {n_sing:>12,} {sing_pct:>8.2f}% {multi_pct:>8.2f}%{flag}")

    print("=" * 65)


def run_split(config_path: pathlib.Path | None = None) -> pd.DataFrame:
    """Run the entity-level split and save the artifact.

    Returns the split DataFrame for downstream use.
    """
    cfg = load_config(config_path) if config_path else load_config()
    train_dir = pathlib.Path(cfg["paths"]["train_dir"])
    artifacts_dir = pathlib.Path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = cfg["split"]
    random_seed: int = cfg["random_seed"]
    fold_sizes: dict[str, float] = split_cfg["fold_sizes"]
    output_path = artifacts_dir / OUTPUT_FILE

    logger.info("Building entity frame from GT ...")
    entity_df = build_entity_frame(train_dir / TRAIN_GROUND_TRUTH)

    logger.info("Running stratified split (seed=%d) ...", random_seed)
    split_df = stratified_split(entity_df, fold_sizes, random_seed)

    # Gate assertion: every entity has a fold
    unassigned = (split_df["fold"] == "").sum()
    if unassigned > 0:
        raise RuntimeError(f"{unassigned} entities were not assigned to a fold -- bug in splitter.")

    print_split_stats(split_df)

    # Gate assertion: singleton ratio in each non-A fold stays within 2pp of fold A
    a_sing_pct = 100.0 * split_df[split_df["fold"] == "A"]["is_singleton"].mean()
    for fold in [f for f in split_df["fold"].unique() if f != "A"]:
        sub = split_df[split_df["fold"] == fold]
        fp = 100.0 * sub["is_singleton"].mean()
        diff = abs(fp - a_sing_pct)
        if diff > 2.5:
            raise RuntimeError(
                f"Fold {fold} singleton% ({fp:.2f}%) deviates {diff:.2f}pp from fold A "
                f"({a_sing_pct:.2f}%) -- exceeds 2.5pp tolerance. Check stratification."
            )
        logger.info("Fold %s singleton%% = %.2f%% (delta vs A = %.2fpp) -- OK", fold, fp, diff)

    output_parquet = artifacts_dir / OUTPUT_FILE
    output_tsv = artifacts_dir / OUTPUT_FILE.replace(".parquet", ".tsv")
    try:
        split_df.to_parquet(output_parquet, index=False)
        saved_path = output_parquet
    except ImportError:
        logger.warning(
            "pyarrow/fastparquet not available -- saving as TSV instead. "
            "Install pyarrow when possible for efficient downstream reads."
        )
        split_df.to_csv(output_tsv, sep="\t", index=False, encoding="utf-8")
        saved_path = output_tsv
    print(f"\nArtifact saved: {saved_path}  ({len(split_df):,} rows)")

    return split_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )
    run_split()
