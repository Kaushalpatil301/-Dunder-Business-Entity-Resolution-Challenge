"""
union.py -- Phase 4 blocking union orchestrator.

Two modes:
  1. --fold-d-only (default): generate candidates only for fold D S1 entities.
     Used for the Phase 4 recall-gate measurement. Fast, memory-safe.
  2. --all-folds: generate for all S1 entities, streaming each channel's
     chunk to disk, then final dedup. Needed before Phase 6 (model training).

Channels (all pandas/numpy -- no sklearn/scipy):
  CH1: exact_sorted           -- exact join on name_sorted
  CH2: exact_expanded_sorted  -- exact join on name_expanded_sorted
  CH3: token_rare             -- 1 rare token (MAX_FREQ=150)
  CH4: token_medium           -- 2 medium-freq tokens (MAX_FREQ=3000)
  CH5: addr_composite         -- 4+-digit number + first street word + country

Phase 4 gate: overall recall on fold D >= 95%.

Outputs (fold-d-only mode):
  artifacts/candidate_pairs_fold_d.tsv   -- gate-measurement artifact
  artifacts/phase4_recall_report.txt     -- gate evidence (printed numbers)
"""

from __future__ import annotations

import logging
import pathlib
import sys
from typing import Any

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
from config.config import load_config
sys.path.insert(0, str(_REPO_ROOT / "code" / "business_entity_resolution" / "src"))
from blocking.fast_normalizer import normalize_for_blocking

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
MIN_KEY_LEN       = 3
MIN_TOK_LEN       = 3
MAX_FREQ_RARE     = 150    # CH3: very discriminative single tokens
MAX_FREQ_MEDIUM   = 3000   # CH4: moderately common tokens, need pair
MIN_SHARED_RARE   = 1
MIN_SHARED_MEDIUM = 2
ADDR_CHUNK        = 100_000
S1_CHUNK          = 200_000


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_tsv(path: pathlib.Path, **kw) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, **kw)
    logger.info("Loaded %s: %d rows", path.name, len(df))
    return df


def explode_gt(gt: pd.DataFrame) -> pd.DataFrame:
    non_empty = gt[gt["matched_entity_ids"].str.strip() != ""].copy()
    non_empty["ml"] = non_empty["matched_entity_ids"].str.split(",")
    pairs = non_empty.explode("ml")[["source1_entity_id", "ml"]].copy()
    pairs.columns = ["s1_id", "target_id"]
    return pairs.reset_index(drop=True)


# --------------------------------------------------------------------------
# Channel helpers
# --------------------------------------------------------------------------

def exact_key_channel(s1n: pd.DataFrame, s2s3n: pd.DataFrame,
                      key: str, ch: str,
                      s1_filter: set[str] | None = None) -> pd.DataFrame:
    sub1 = s1n[["entity_id", key]].rename(columns={"entity_id": "s1_id"})
    if s1_filter is not None:
        sub1 = sub1[sub1["s1_id"].isin(s1_filter)]
    sub1 = sub1[sub1[key].str.len() >= MIN_KEY_LEN]
    sub2 = s2s3n[["entity_id", key]].rename(columns={"entity_id": "target_id"})
    sub2 = sub2[sub2[key].str.len() >= MIN_KEY_LEN]
    m = pd.merge(sub1, sub2, on=key)[["s1_id", "target_id"]].drop_duplicates()
    m = m[m["s1_id"] != m["target_id"]]
    m["channel"] = ch
    logger.info("CH %s: %d pairs", ch, len(m))
    return m


def build_token_table(s2s3n: pd.DataFrame, max_freq: int,
                      label: str) -> pd.DataFrame:
    sub = s2s3n[["entity_id", "name_alphanum"]].copy()
    sub = sub[sub["name_alphanum"].str.strip() != ""]
    tok = sub.assign(token=sub["name_alphanum"].str.split()).explode("token")
    tok = tok[tok["token"].str.len() >= MIN_TOK_LEN].copy()
    freq = tok.groupby("token")["entity_id"].nunique()
    keep = freq[freq <= max_freq].index
    tok = tok[tok["token"].isin(keep)][["token", "entity_id"]].drop_duplicates()
    logger.info("Token table %s: %d rows (%d unique tokens)", label, len(tok), len(keep))
    return tok


def token_channel(s1n: pd.DataFrame, tok_table: pd.DataFrame,
                  min_shared: int, ch: str,
                  s1_filter: set[str] | None = None) -> pd.DataFrame:
    valid = set(tok_table["token"].unique())
    s1_sub = s1n[["entity_id", "name_alphanum"]].copy()
    if s1_filter is not None:
        s1_sub = s1_sub[s1_sub["entity_id"].isin(s1_filter)]
    s1_sub = s1_sub[s1_sub["name_alphanum"].str.strip() != ""]

    parts: list[pd.DataFrame] = []
    n_chunks = max(1, (len(s1_sub) + S1_CHUNK - 1) // S1_CHUNK)
    for ci in range(n_chunks):
        chunk = s1_sub.iloc[ci * S1_CHUNK : (ci + 1) * S1_CHUNK]
        s1_tok = chunk.assign(token=chunk["name_alphanum"].str.split()).explode("token")
        s1_tok = s1_tok[s1_tok["token"].str.len() >= MIN_TOK_LEN]
        s1_tok = s1_tok[s1_tok["token"].isin(valid)]
        s1_tok = s1_tok.rename(columns={"entity_id": "s1_id"})
        if s1_tok.empty:
            continue
        m = pd.merge(s1_tok[["s1_id", "token"]],
                     tok_table.rename(columns={"entity_id": "target_id"}), on="token")
        m = m[m["s1_id"] != m["target_id"]]
        if min_shared > 1:
            cnt = m.groupby(["s1_id", "target_id"]).size().reset_index(name="n")
            cnt = cnt[cnt["n"] >= min_shared][["s1_id", "target_id"]]
        else:
            cnt = m[["s1_id", "target_id"]].drop_duplicates()
        parts.append(cnt)
        logger.info("  %s chunk %d/%d", ch, ci + 1, n_chunks)

    if not parts:
        return pd.DataFrame(columns=["s1_id", "target_id", "channel"])
    result = pd.concat(parts, ignore_index=True).drop_duplicates()
    result["channel"] = ch
    logger.info("CH %s: %d pairs", ch, len(result))
    return result


def _addr_key(addr_alnum: pd.Series) -> pd.Series:
    ext = addr_alnum.str.extract(r"(?<!\d)(\d{4,})\s+([a-z]{4,})", expand=True)
    key = ext[0].fillna("") + " " + ext[1].fillna("")
    return key.str.strip()


def address_channel(s1n: pd.DataFrame, s2s3n: pd.DataFrame,
                    s1_filter: set[str] | None = None) -> pd.DataFrame:
    s1c = s1n.copy(); s1c["addr_key"] = _addr_key(s1c["addr_alphanum"])
    s2c = s2s3n.copy(); s2c["addr_key"] = _addr_key(s2c["addr_alphanum"])
    s1_sub = s1c[["entity_id", "addr_key", "country"]]
    if s1_filter is not None:
        s1_sub = s1_sub[s1_sub["entity_id"].isin(s1_filter)]
    s1_sub = s1_sub[s1_sub["addr_key"].str.len() >= 8]
    s2_sub = s2c[["entity_id", "addr_key", "country"]]
    s2_sub = s2_sub[s2_sub["addr_key"].str.len() >= 8]

    parts: list[pd.DataFrame] = []
    n_ch = max(1, (len(s1_sub) + ADDR_CHUNK - 1) // ADDR_CHUNK)
    for ci in range(n_ch):
        chunk = s1_sub.iloc[ci * ADDR_CHUNK : (ci + 1) * ADDR_CHUNK]
        m = pd.merge(chunk.rename(columns={"entity_id": "s1_id"}),
                     s2_sub.rename(columns={"entity_id": "target_id"}),
                     on=["addr_key", "country"])
        m = m[m["s1_id"] != m["target_id"]][["s1_id", "target_id"]].drop_duplicates()
        parts.append(m)
    if not parts:
        return pd.DataFrame(columns=["s1_id", "target_id", "channel"])
    result = pd.concat(parts, ignore_index=True).drop_duplicates()
    result["channel"] = "addr_composite"
    logger.info("CH addr_composite: %d pairs", len(result))
    return result


# --------------------------------------------------------------------------
# Recall measurement
# --------------------------------------------------------------------------

def measure_recall(cands: pd.DataFrame, true_pairs: pd.DataFrame,
                   fold_d_ids: set[str]) -> dict[str, Any]:
    tp_d   = true_pairs[true_pairs["s1_id"].isin(fold_d_ids)]
    tp_set = set(zip(tp_d["s1_id"], tp_d["target_id"]))
    n_true = len(tp_set)

    cand_d   = cands[cands["s1_id"].isin(fold_d_ids)]
    cand_set = set(zip(cand_d["s1_id"], cand_d["target_id"]))
    n_found  = len(tp_set & cand_set)

    per_ch: dict[str, float] = {}
    for ch, grp in cand_d.groupby("channel"):
        ch_set = set(zip(grp["s1_id"], grp["target_id"]))
        per_ch[ch] = len(tp_set & ch_set) / n_true if n_true > 0 else 0.0

    return {
        "overall_recall": n_found / n_true if n_true > 0 else 0.0,
        "per_channel": per_ch,
        "n_true_fold_d": n_true,
        "n_found": n_found,
        "n_cands_fold_d": len(cand_set),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run_blocking(fold_d_only: bool = True) -> None:
    """
    fold_d_only=True  -> recall gate measurement only (memory-safe).
    fold_d_only=False -> full candidate generation for all S1 entities.
                         WARNING: may require 20-30 GB RAM or streaming output.
    """
    cfg       = load_config()
    train_dir = pathlib.Path(cfg["paths"]["train_dir"])
    artifacts = pathlib.Path(cfg["paths"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    def emit(msg: str = "") -> None:
        lines.append(msg)
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="backslashreplace").decode("ascii"))

    mode = "FOLD-D RECALL GATE" if fold_d_only else "FULL CANDIDATE GENERATION"
    emit("=" * 70)
    emit(f"PHASE 4 BLOCKING UNION  [{mode}]")
    emit("=" * 70)

    emit("\n[1] Loading + normalizing ...")
    s1    = load_tsv(train_dir / "train_source1.tsv")
    s2    = load_tsv(train_dir / "train_source2.tsv")
    s3    = load_tsv(train_dir / "train_source3.tsv")
    gt    = load_tsv(train_dir / "train_ground_truth.tsv")
    emit("  Normalizing S1 ...")
    s1n   = normalize_for_blocking(s1)
    emit("  Normalizing S2 ...")
    s2n   = normalize_for_blocking(s2)
    emit("  Normalizing S3 ...")
    s3n   = normalize_for_blocking(s3)
    s2s3n = pd.concat([s2n, s3n], ignore_index=True)
    emit(f"  S2+S3: {len(s2s3n):,} records")

    split_df   = pd.read_csv(artifacts / "entity_split.tsv", sep="\t", dtype=str)
    fold_d     = set(split_df[split_df["fold"] == "D"]["source1_entity_id"])
    true_pairs = explode_gt(gt)
    s1_filter  = fold_d if fold_d_only else None
    emit(f"  Fold D: {len(fold_d):,} | GT pairs: {len(true_pairs):,}")
    emit(f"  S1 entities to process: {len(s1_filter) if s1_filter else len(s1n):,}")

    # --- Build token tables once (shared across channels) ---
    emit("\n[2] Building token tables ...")
    tok_rare   = build_token_table(s2s3n, MAX_FREQ_RARE,   "rare")
    tok_medium = build_token_table(s2s3n, MAX_FREQ_MEDIUM, "medium")

    # --- Run channels with s1_filter ---
    all_cands: list[pd.DataFrame] = []

    emit("\n[3] CH1: exact_sorted ...")
    ch1 = exact_key_channel(s1n, s2s3n, "name_sorted", "exact_sorted", s1_filter)
    all_cands.append(ch1); emit(f"  -> {len(ch1):,}")

    emit("\n[4] CH2: exact_expanded_sorted ...")
    ch2 = exact_key_channel(s1n, s2s3n, "name_expanded_sorted",
                            "exact_expanded_sorted", s1_filter)
    all_cands.append(ch2); emit(f"  -> {len(ch2):,}")

    emit("\n[5] CH3: token_rare (min_shared=1, max_freq=150) ...")
    ch3 = token_channel(s1n, tok_rare, MIN_SHARED_RARE, "token_rare", s1_filter)
    all_cands.append(ch3); emit(f"  -> {len(ch3):,}")

    emit("\n[6] CH4: token_medium (min_shared=2, max_freq=3000) ...")
    ch4 = token_channel(s1n, tok_medium, MIN_SHARED_MEDIUM, "token_medium", s1_filter)
    all_cands.append(ch4); emit(f"  -> {len(ch4):,}")

    emit("\n[7] CH5: addr_composite ...")
    ch5 = address_channel(s1n, s2s3n, s1_filter)
    all_cands.append(ch5); emit(f"  -> {len(ch5):,}")

    emit("\n[8] Union + dedup ...")
    union_df = pd.concat(all_cands, ignore_index=True)
    n_raw    = len(union_df)
    union_df = union_df.drop_duplicates(subset=["s1_id", "target_id"], keep="first")
    emit(f"  Raw: {n_raw:,}  -> After dedup: {len(union_df):,}")

    emit("\n[9] Recall (fold D) ...")
    stats = measure_recall(union_df, true_pairs, fold_d)

    emit("\n" + "=" * 70)
    emit("PHASE 4 RECALL REPORT (fold D held-out)")
    emit("=" * 70)
    emit(f"  True pairs in fold D:     {stats['n_true_fold_d']:>12,}")
    emit(f"  True pairs found:         {stats['n_found']:>12,}")
    emit(f"  Candidates (fold D):      {stats['n_cands_fold_d']:>12,}")
    emit(f"  OVERALL RECALL:           {stats['overall_recall']:>11.4f}  ({stats['overall_recall']*100:.2f}%)")
    emit()
    emit("  Per-channel recall (independent contribution from each channel):")
    for ch, r in sorted(stats["per_channel"].items(), key=lambda x: -x[1]):
        emit(f"    {ch:<30}: {r:.4f}  ({r*100:.2f}%)")
    emit("=" * 70)
    if stats["overall_recall"] >= 0.95:
        emit("\nPHASE 4 GATE: PASS  (recall >= 95%)")
    else:
        emit(f"\nPHASE 4 GATE: FAIL  ({stats['overall_recall']*100:.2f}% < 95%)")
        emit("  Missing pairs: investigate per-channel recall above.")

    emit("\n[10] Writing artifacts ...")
    out = union_df.rename(columns={"s1_id": "source1_entity_id",
                                   "target_id": "candidate_entity_id"})
    suffix    = "fold_d" if fold_d_only else "v1"
    cand_path = artifacts / f"candidate_pairs_{suffix}.tsv"
    out.to_csv(cand_path, sep="\t", index=False, encoding="utf-8")
    emit(f"  Saved: {cand_path}  ({len(out):,} rows)")

    report = artifacts / "phase4_recall_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    emit(f"  Saved: {report}")
    emit("\n" + "=" * 70)
    emit("END PHASE 4")
    emit("=" * 70)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s",
                        stream=sys.stderr)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    p = argparse.ArgumentParser()
    p.add_argument("--all-folds", action="store_true",
                   help="Generate for all S1 entities (not just fold D).")
    args = p.parse_args()
    run_blocking(fold_d_only=not args.all_folds)
