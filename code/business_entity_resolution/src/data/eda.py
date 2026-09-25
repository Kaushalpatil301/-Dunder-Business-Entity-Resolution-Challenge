"""
eda.py -- Phase 1 EDA for the Amazon ML Challenge 2026.

Loads train_source{1,2,3}.tsv and train_ground_truth.tsv, then computes and
prints:
  1. Record counts per source
  2. Country split (train)
  3. Match cardinality distribution (singleton%, 1-match%, multi-match%)
  4. S2/S3 target-side uniqueness check (gates linear_sum_assignment eligibility)
  5. Representative noisy-true-pair sample (up to SAMPLE_SIZE rows)

All results are printed to stdout (Phase 1 gate evidence) and saved to
artifacts/phase1_eda_summary.txt so they can be reproduced without re-running.

Assumptions:
  - Raw TSVs must be read with sep="\t" (a missing sep silently collapses rows).
  - Raw files under dataset/ are never written to.
  - country is treated as an open string -- no hardcoded {US, India} branching.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
from typing import Dict, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to repo root -- run from repo root)
# ---------------------------------------------------------------------------
TRAIN_DIR = pathlib.Path("dataset/train")
ARTIFACTS_DIR = pathlib.Path("artifacts")
SUMMARY_FILE = ARTIFACTS_DIR / "phase1_eda_summary.txt"

SAMPLE_SIZE = 25  # noisy-true-pair sample rows to show


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_source(path: pathlib.Path) -> pd.DataFrame:
    """Load a source TSV with sep=TAB.

    Assumption: file has columns entity_id, business_name, business_address, country.
    Always uses sep='\\t' -- a missing sep silently collapses rows into one column.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    logger.info("Loaded %s: %d rows, columns=%s", path.name, len(df), list(df.columns))
    return df


def load_ground_truth(path: pathlib.Path) -> pd.DataFrame:
    """Load train_ground_truth.tsv with sep=TAB.

    Assumption: columns are source1_entity_id, matched_entity_ids (comma-separated,
    empty string = singleton).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    logger.info("Loaded %s: %d rows, columns=%s", path.name, len(df), list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def count_records(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame) -> Dict[str, int]:
    """Return record counts per source.

    No recall/precision assumptions -- purely a count.
    """
    return {"S1": len(s1), "S2": len(s2), "S3": len(s3)}


def country_split(s1: pd.DataFrame) -> pd.Series:
    """Return country value counts (absolute + %) from S1.

    S1 is the reference set; country distribution here drives train-time coverage.
    country is treated as an open string -- no hardcoded set assumed.
    """
    return s1["country"].value_counts(dropna=False)


def match_cardinality(gt: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float]]:
    """Compute per-S1-entity match cardinality and return distribution stats.

    A singleton has matched_entity_ids == '' (empty string).
    1-match has exactly one comma-less token.
    multi-match has one or more commas.

    Returns:
        (cardinality_series, stats_dict) where stats_dict has:
            singleton_pct, one_match_pct, multi_match_pct, max_matches, mean_matches
    """
    # Parse match count per entity
    def _count(val: str) -> int:
        if val.strip() == "":
            return 0
        return len(val.split(","))

    gt = gt.copy()
    gt["n_matches"] = gt["matched_entity_ids"].apply(_count)
    n_total = len(gt)

    singleton_n = (gt["n_matches"] == 0).sum()
    one_n = (gt["n_matches"] == 1).sum()
    multi_n = (gt["n_matches"] > 1).sum()

    stats = {
        "singleton_pct": 100.0 * singleton_n / n_total,
        "one_match_pct": 100.0 * one_n / n_total,
        "multi_match_pct": 100.0 * multi_n / n_total,
        "max_matches": int(gt["n_matches"].max()),
        "mean_matches": float(gt["n_matches"].mean()),
        "total_entities": n_total,
        "singleton_n": int(singleton_n),
        "one_n": int(one_n),
        "multi_n": int(multi_n),
    }
    return gt["n_matches"].value_counts().sort_index(), stats


def target_side_uniqueness(gt: pd.DataFrame) -> Dict[str, object]:
    """Check whether S2/S3 IDs appear more than once on the target side of GT.

    This gates linear_sum_assignment eligibility per architecture.md invariant 6:
    forced 1:1 matching is only eligible if target-side uniqueness holds.

    Returns a dict with:
        s2_unique: bool -- True if every S2 ID appears in GT at most once
        s3_unique: bool -- True if every S3 ID appears in GT at most once
        s2_total_ids: int
        s3_total_ids: int
        s2_duplicated_ids: int -- count of S2 IDs appearing more than once
        s3_duplicated_ids: int
    """
    all_ids: list[str] = []
    for row in gt["matched_entity_ids"]:
        if row.strip():
            all_ids.extend(row.split(","))

    s2_ids = [i for i in all_ids if i.startswith("S2-")]
    s3_ids = [i for i in all_ids if i.startswith("S3-")]

    s2_series = pd.Series(s2_ids)
    s3_series = pd.Series(s3_ids)

    s2_dup = int((s2_series.value_counts() > 1).sum())
    s3_dup = int((s3_series.value_counts() > 1).sum())

    return {
        "s2_unique": s2_dup == 0,
        "s3_unique": s3_dup == 0,
        "s2_total_ids": len(s2_ids),
        "s3_total_ids": len(s3_ids),
        "s2_duplicated_ids": s2_dup,
        "s3_duplicated_ids": s3_dup,
    }


def noisy_pair_sample(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: pd.DataFrame,
    n: int = SAMPLE_SIZE,
) -> pd.DataFrame:
    """Return a sample of true matched pairs for Phase 3 normalization spot-check.

    Samples from non-singleton GT rows, joins S1 and target (S2 or S3) records
    side-by-side so name/address noise is visible at a glance.

    Assumption: sampling is deterministic via random_state=42.
    """
    non_singleton = gt[gt["matched_entity_ids"].str.strip() != ""].copy()
    # Explode to one row per matched pair
    non_singleton["matched_list"] = non_singleton["matched_entity_ids"].str.split(",")
    pairs = non_singleton.explode("matched_list")[["source1_entity_id", "matched_list"]]
    pairs.columns = ["s1_id", "target_id"]
    pairs = pairs.sample(min(n, len(pairs)), random_state=42).reset_index(drop=True)

    s1_lookup = s1.set_index("entity_id")[["business_name", "business_address", "country"]]
    s2_lookup = s2.set_index("entity_id")[["business_name", "business_address"]]
    s3_lookup = s3.set_index("entity_id")[["business_name", "business_address"]]
    target_lookup = pd.concat([s2_lookup, s3_lookup])

    pairs["s1_name"] = pairs["s1_id"].map(s1_lookup["business_name"])
    pairs["s1_addr"] = pairs["s1_id"].map(s1_lookup["business_address"])
    pairs["s1_country"] = pairs["s1_id"].map(s1_lookup["country"])
    pairs["tgt_name"] = pairs["target_id"].map(target_lookup["business_name"])
    pairs["tgt_addr"] = pairs["target_id"].map(target_lookup["business_address"])
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eda(train_dir: pathlib.Path = TRAIN_DIR) -> None:
    """Run all Phase 1 EDA steps and print + save results.

    Gate evidence: every line printed here is the authoritative record.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    emit("=" * 70)
    emit("PHASE 1 EDA -- Amazon ML Challenge 2026")
    emit("=" * 70)

    # --- Load ---
    emit("\n[1] Loading files ...")
    s1 = load_source(train_dir / "train_source1.tsv")
    s2 = load_source(train_dir / "train_source2.tsv")
    s3 = load_source(train_dir / "train_source3.tsv")
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")

    # --- Record counts ---
    emit("\n[2] RECORD COUNTS")
    counts = count_records(s1, s2, s3)
    for src, n in counts.items():
        emit(f"  {src}: {n:,} records")

    # --- Country split ---
    emit("\n[3] COUNTRY SPLIT (S1 -- reference set)")
    cs = country_split(s1)
    total_s1 = len(s1)
    for country, cnt in cs.items():
        emit(f"  {country!r:15s}: {cnt:>8,}  ({100.0 * cnt / total_s1:.2f}%)")

    # --- Match cardinality ---
    emit("\n[4] MATCH CARDINALITY DISTRIBUTION (from train_ground_truth.tsv)")
    _, stats = match_cardinality(gt)
    emit(f"  Total S1 entities in GT: {stats['total_entities']:,}")
    emit(f"  Singletons (0 matches):  {stats['singleton_n']:>8,}  ({stats['singleton_pct']:.2f}%)")
    emit(f"  1-match:                 {stats['one_n']:>8,}  ({stats['one_match_pct']:.2f}%)")
    emit(f"  Multi-match (>1):        {stats['multi_n']:>8,}  ({stats['multi_match_pct']:.2f}%)")
    emit(f"  Max matches (one entity):{stats['max_matches']:>8,}")
    emit(f"  Mean matches per entity: {stats['mean_matches']:>11.4f}")

    # --- Target-side uniqueness ---
    emit("\n[5] TARGET-SIDE UNIQUENESS CHECK (gates linear_sum_assignment eligibility)")
    uniq = target_side_uniqueness(gt)
    emit(f"  S2 IDs in GT: {uniq['s2_total_ids']:,}  |  duplicated: {uniq['s2_duplicated_ids']:,}  |  unique={uniq['s2_unique']}")
    emit(f"  S3 IDs in GT: {uniq['s3_total_ids']:,}  |  duplicated: {uniq['s3_duplicated_ids']:,}  |  unique={uniq['s3_unique']}")
    if uniq["s2_unique"] and uniq["s3_unique"]:
        emit("  -> TARGET-SIDE UNIQUENESS HOLDS for both S2 and S3.")
        emit("     linear_sum_assignment is ELIGIBLE pending error-analysis in Phase 8.")
    else:
        emit("  -> TARGET-SIDE UNIQUENESS DOES NOT HOLD (at least one source has duplicates).")
        emit("     linear_sum_assignment is NOT eligible per architecture.md invariant 6.")

    # --- Noisy-pair sample ---
    emit(f"\n[6] NOISY TRUE-PAIR SAMPLE (up to {SAMPLE_SIZE} rows -- for Phase 3 spot-check)")
    sample = noisy_pair_sample(s1, s2, s3, gt)
    pd.set_option("display.max_colwidth", 55)
    pd.set_option("display.width", 200)
    sample_str = sample[["s1_id", "target_id", "s1_country", "s1_name", "tgt_name", "s1_addr", "tgt_addr"]].to_string(index=False)
    lines.append(sample_str)  # store full UTF-8 in artifact

    emit("\n" + "=" * 70)
    emit("END OF PHASE 1 EDA")
    emit("=" * 70)

    # Save artifact first (UTF-8, full Unicode preserved)
    SUMMARY_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nArtifact saved: {SUMMARY_FILE}")

    # Print sample to console with ascii-escaped fallback (safe on cp1252 Windows terminals)
    safe_sample = sample_str.encode("ascii", errors="backslashreplace").decode("ascii")
    print(safe_sample)


if __name__ == "__main__":
    # Force UTF-8 on stdout so Indic / non-Latin characters in business
    # names and addresses don't crash the Windows cp1252 console.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )
    run_eda()
