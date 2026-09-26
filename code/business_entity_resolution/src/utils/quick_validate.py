#!/usr/bin/env python
"""
quick_validate.py — Quick sanity check on output files.

Runs:
1. utils/validate_submission.py
2. Counts singletons vs matched entities
3. Shows top-10 most-matched S1 entities

Run from repo root:
    python quick_validate.py
"""

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent

import pandas as pd

matching  = REPO / "output" / "matching_results.tsv"
candidate = REPO / "output" / "candidate_pairs.tsv"
test_dir  = REPO / "dataset" / "test"

print("=" * 60)
print("QUICK VALIDATION")
print("=" * 60)

# 1. Format validator
print("\n[1] Running validate_submission.py ...")
result = subprocess.run(
    [sys.executable, str(REPO / "utils" / "validate_submission.py"),
     "--matching",  str(matching),
     "--candidate", str(candidate),
     "--test-dir",  str(test_dir)],
    capture_output=True, text=True,
)
print(result.stdout)
if result.stderr.strip():
    print(result.stderr)
print(f"Exit code: {result.returncode} ({'PASS' if result.returncode == 0 else 'FAIL'})")

# 2. Stats on matching_results
print("\n[2] matching_results.tsv stats:")
df = pd.read_csv(matching, sep="\t", dtype=str, keep_default_na=False)
print(f"  Total rows:      {len(df):,}")
n_sing = (df["matched_entity_ids"] == "").sum()
n_mat  = len(df) - n_sing
print(f"  Singletons:      {n_sing:,}  ({100*n_sing/len(df):.1f}%)")
print(f"  Matched:         {n_mat:,}  ({100*n_mat/len(df):.1f}%)")

df["n_matches"] = df["matched_entity_ids"].apply(
    lambda x: len(x.split(",")) if x.strip() else 0
)
print(f"  Mean matches per entity (matched only): "
      f"{df[df['n_matches']>0]['n_matches'].mean():.2f}")
print(f"  Max matches: {df['n_matches'].max()}")

print("\n  Top 10 most-matched S1 entities:")
print(df.nlargest(10, "n_matches")[["source1_entity_id", "n_matches", "matched_entity_ids"]].to_string(index=False))

# 3. Candidate stats
if candidate.exists():
    print("\n[3] candidate_pairs.tsv stats:")
    cd = pd.read_csv(candidate, sep="\t", dtype=str, keep_default_na=False)
    cd["n_cands"] = cd["candidate_entity_ids"].apply(
        lambda x: len(x.split(",")) if x.strip() else 0
    )
    print(f"  Total rows:      {len(cd):,}")
    print(f"  Mean candidates: {cd['n_cands'].mean():.2f}")
    total_cands = cd["n_cands"].sum()
    print(f"  Total candidates:{total_cands:,}")

print("\n" + "=" * 60)
