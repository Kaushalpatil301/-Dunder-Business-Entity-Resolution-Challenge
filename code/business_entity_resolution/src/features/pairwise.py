"""
pairwise.py -- Phase 5: Pairwise feature extraction for candidate pairs.

Reads a candidate TSV (source1_entity_id, candidate_entity_id, [channel])
and the normalized source files, then produces a feature matrix saved as
a TSV with all feature columns plus a binary `label` column when GT is
provided (train mode) or without `label` (inference mode).

Features (all country-agnostic, safe for France):
  name_token_sort_ratio    -- RapidFuzz token_sort_ratio on name_alphanum
  name_partial_ratio       -- RapidFuzz partial_ratio on name_alphanum
  name_expanded_sort_ratio -- RapidFuzz token_sort_ratio on name_expanded
  name_sorted_exact        -- 1 if name_sorted views are identical
  name_alphanum_exact      -- 1 if name_alphanum views are identical
  addr_token_jaccard       -- Jaccard of address alphanum token sets
  addr_numeric_match       -- 1 if first numeric in address matches
  country_match            -- 1 if country labels are the same string
  name_len_ratio           -- min/max of character lengths of business_name
  name_token_len_ratio     -- min/max of token count of name_alphanum

Usage (train):
  python src/features/pairwise.py \\
      --candidates artifacts/candidate_pairs_fold_a.tsv \\
      --input-dir dataset/train \\
      --gt dataset/train/train_ground_truth.tsv \\
      --output artifacts/features_fold_a.tsv

Usage (inference / test):
  python src/features/pairwise.py \\
      --candidates output/candidate_pairs.tsv \\
      --source1 dataset/test/test_source1.tsv \\
      --source2 dataset/test/test_source2.tsv \\
      --source3 dataset/test/test_source3.tsv \\
      --output artifacts/features_test.tsv
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz as rf_fuzz
    from rapidfuzz import process as rf_process
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# --------------------------------------------------------------------------
# Bootstrap sys.path so we can import blocking/fast_normalizer
# --------------------------------------------------------------------------
_SRC   = pathlib.Path(__file__).resolve().parents[1]
_REPO  = pathlib.Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

from blocking.fast_normalizer import normalize_for_blocking

logger = logging.getLogger(__name__)

CHUNK = 200_000   # rows processed per batch


# --------------------------------------------------------------------------
# Loader helpers
# --------------------------------------------------------------------------

def _load_tsv(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    logger.info("Loaded %s: %d rows", path.name, len(df))
    return df


def _build_lookup(norm_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by entity_id with normalised view columns."""
    cols = [
        "entity_id", "business_name", "country",
        "name_alphanum", "name_sorted", "name_expanded", "name_expanded_sorted",
        "addr_alphanum", "addr_numeric",
    ]
    return norm_df[cols].set_index("entity_id")


# --------------------------------------------------------------------------
# Feature computation (vectorized / row-batch)
# --------------------------------------------------------------------------

def _safe_ratio(fn, a_ser: pd.Series, b_ser: pd.Series) -> pd.Series:
    """Apply a RapidFuzz ratio function element-wise, return 0.0–1.0 Series."""
    if _HAS_RAPIDFUZZ:
        scores = [
            fn(str(a), str(b)) / 100.0 if (a and b) else 0.0
            for a, b in zip(a_ser, b_ser)
        ]
    else:
        # Fallback: character-level overlap fraction
        scores = []
        for a, b in zip(a_ser, b_ser):
            sa, sb = set(str(a)), set(str(b))
            d = len(sa & sb) / max(len(sa | sb), 1)
            scores.append(d)
    return pd.Series(scores, index=a_ser.index, dtype="float32")


def _jaccard(a_ser: pd.Series, b_ser: pd.Series) -> pd.Series:
    """Token-level Jaccard similarity."""
    def _j(a, b):
        ta, tb = set(str(a).split()), set(str(b).split())
        if not ta and not tb:
            return 1.0
        union = len(ta | tb)
        return len(ta & tb) / union if union else 0.0
    return pd.Series([_j(a, b) for a, b in zip(a_ser, b_ser)],
                     index=a_ser.index, dtype="float32")


def _len_ratio(a_ser: pd.Series, b_ser: pd.Series) -> pd.Series:
    a_len = a_ser.str.len().clip(lower=1)
    b_len = b_ser.str.len().clip(lower=1)
    ratio = np.minimum(a_len, b_len) / np.maximum(a_len, b_len)
    return ratio.astype("float32")


def _tok_len_ratio(a_ser: pd.Series, b_ser: pd.Series) -> pd.Series:
    a_n = a_ser.str.split().str.len().clip(lower=1)
    b_n = b_ser.str.split().str.len().clip(lower=1)
    ratio = np.minimum(a_n, b_n) / np.maximum(a_n, b_n)
    return ratio.astype("float32")


def compute_features(pairs: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """
    pairs: DataFrame with columns ['s1_id', 'target_id']
    lookup: entity_id-indexed DataFrame of normalised views
    Returns: feature DataFrame (same index as pairs)
    """
    # --- Join normalised views ---
    s1 = lookup.reindex(pairs["s1_id"].values)
    tgt = lookup.reindex(pairs["target_id"].values)

    # Reset indices for vectorized ops
    s1.index  = pairs.index
    tgt.index = pairs.index

    feat = pd.DataFrame(index=pairs.index)

    # ---- Name similarity features ----
    if _HAS_RAPIDFUZZ:
        feat["name_token_sort_ratio"]    = _safe_ratio(rf_fuzz.token_sort_ratio,
                                                       s1["name_alphanum"], tgt["name_alphanum"])
        feat["name_partial_ratio"]       = _safe_ratio(rf_fuzz.partial_ratio,
                                                       s1["name_alphanum"], tgt["name_alphanum"])
        feat["name_expanded_sort_ratio"] = _safe_ratio(rf_fuzz.token_sort_ratio,
                                                       s1["name_expanded"], tgt["name_expanded"])
    else:
        feat["name_token_sort_ratio"]    = _jaccard(s1["name_alphanum"], tgt["name_alphanum"])
        feat["name_partial_ratio"]       = _jaccard(s1["name_alphanum"], tgt["name_alphanum"])
        feat["name_expanded_sort_ratio"] = _jaccard(s1["name_expanded"], tgt["name_expanded"])

    feat["name_sorted_exact"]   = (s1["name_sorted"].values == tgt["name_sorted"].values).astype("float32")
    feat["name_alphanum_exact"] = (s1["name_alphanum"].values == tgt["name_alphanum"].values).astype("float32")

    # ---- Address features ----
    feat["addr_token_jaccard"] = _jaccard(s1["addr_alphanum"], tgt["addr_alphanum"])
    feat["addr_numeric_match"] = (
        (s1["addr_numeric"].values == tgt["addr_numeric"].values) &
        (s1["addr_numeric"].str.len() >= 2).values
    ).astype("float32")

    # ---- Country match ----
    feat["country_match"] = (s1["country"].values == tgt["country"].values).astype("float32")

    # ---- Length ratios ----
    feat["name_len_ratio"]       = _len_ratio(s1["name_alphanum"], tgt["name_alphanum"])
    feat["name_token_len_ratio"] = _tok_len_ratio(s1["name_alphanum"], tgt["name_alphanum"])

    return feat


# --------------------------------------------------------------------------
# GT loading for train mode
# --------------------------------------------------------------------------

def build_gt_lookup(gt_path: pathlib.Path) -> dict[str, frozenset]:
    """Return {s1_id: frozenset(matched_ids)} including singletons as empty set."""
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    result: dict[str, frozenset] = {}
    for _, row in gt.iterrows():
        s1 = row["source1_entity_id"]
        raw = row["matched_entity_ids"].strip()
        matches = frozenset(raw.split(",")) if raw else frozenset()
        result[s1] = matches
    return result


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(
    candidates_path: pathlib.Path,
    output_path: pathlib.Path,
    source1_path: pathlib.Path,
    source2_path: pathlib.Path,
    source3_path: pathlib.Path,
    gt_path: pathlib.Path | None = None,
    chunk_size: int = CHUNK,
) -> None:
    logger.info("=== Phase 5: Pairwise Feature Extraction ===")
    if not _HAS_RAPIDFUZZ:
        logger.warning("rapidfuzz not found — using Jaccard fallback (slower/weaker).")

    # --- Load and normalize sources ---
    s1  = _load_tsv(source1_path)
    s2  = _load_tsv(source2_path)
    s3  = _load_tsv(source3_path)
    s23 = pd.concat([s2, s3], ignore_index=True)

    logger.info("Normalizing sources ...")
    s1n  = normalize_for_blocking(s1)
    s23n = normalize_for_blocking(s23)

    lookup = _build_lookup(pd.concat([s1n, s23n], ignore_index=True))

    # --- Load GT if provided ---
    gt_lookup: dict[str, frozenset] | None = None
    if gt_path is not None:
        gt_lookup = build_gt_lookup(gt_path)
        logger.info("GT loaded: %d S1 entities", len(gt_lookup))

    # --- Load candidates ---
    cands = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False)
    logger.info("Candidates: %d rows", len(cands))

    # Normalise column names to s1_id / target_id
    if "source1_entity_id" in cands.columns and "candidate_entity_id" in cands.columns:
        cands = cands.rename(columns={
            "source1_entity_id": "s1_id",
            "candidate_entity_id": "target_id",
        })
    elif "s1_id" not in cands.columns:
        raise ValueError(f"Cannot detect id columns in {candidates_path}. "
                         f"Expected source1_entity_id+candidate_entity_id or s1_id+target_id.")

    # --- Chunk processing ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True
    n_written = 0
    n_chunks = max(1, (len(cands) + chunk_size - 1) // chunk_size)

    for ci in range(n_chunks):
        chunk = cands.iloc[ci * chunk_size: (ci + 1) * chunk_size].copy()
        chunk = chunk.reset_index(drop=True)

        # Skip pairs whose IDs aren't in lookup (IDs from different split sources)
        valid_mask = chunk["s1_id"].isin(lookup.index) & chunk["target_id"].isin(lookup.index)
        chunk = chunk[valid_mask].reset_index(drop=True)
        if chunk.empty:
            continue

        feats = compute_features(chunk, lookup)
        feats.insert(0, "s1_id",     chunk["s1_id"].values)
        feats.insert(1, "target_id", chunk["target_id"].values)

        if gt_lookup is not None:
            labels = []
            for _, row in chunk.iterrows():
                s1_id = row["s1_id"]
                tgt_id = row["target_id"]
                matched = gt_lookup.get(s1_id, frozenset())
                labels.append(1 if tgt_id in matched else 0)
            feats["label"] = labels

        feats.to_csv(
            output_path, sep="\t", index=False,
            header=first_chunk, mode="w" if first_chunk else "a",
            encoding="utf-8",
        )
        first_chunk = False
        n_written += len(feats)
        logger.info("  Chunk %d/%d done: %d rows written (total %d)",
                    ci + 1, n_chunks, len(feats), n_written)

    logger.info("Feature file written: %s (%d rows)", output_path, n_written)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 5: Pairwise feature extraction")
    p.add_argument("--candidates",  required=True,  type=pathlib.Path)
    p.add_argument("--output",      required=True,  type=pathlib.Path)
    p.add_argument("--source1",     type=pathlib.Path)
    p.add_argument("--source2",     type=pathlib.Path)
    p.add_argument("--source3",     type=pathlib.Path)
    p.add_argument("--input-dir",   type=pathlib.Path,
                   help="Shortcut: load train_source{1,2,3}.tsv from this dir")
    p.add_argument("--gt",          type=pathlib.Path,
                   help="Ground truth TSV (train mode only)")
    p.add_argument("--chunk",       type=int, default=CHUNK)
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

    if args.input_dir:
        d = args.input_dir
        prefix = "train" if (d / "train_source1.tsv").exists() else "test"
        s1  = d / f"{prefix}_source1.tsv"
        s2  = d / f"{prefix}_source2.tsv"
        s3  = d / f"{prefix}_source3.tsv"
    else:
        if not all([args.source1, args.source2, args.source3]):
            raise SystemExit("Provide --input-dir OR --source1/2/3")
        s1, s2, s3 = args.source1, args.source2, args.source3

    run(
        candidates_path=args.candidates,
        output_path=args.output,
        source1_path=s1,
        source2_path=s2,
        source3_path=s3,
        gt_path=args.gt,
        chunk_size=args.chunk,
    )
