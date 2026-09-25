# Business Entity Resolution — Reproduction Guide

## Overview
This folder contains the complete ML pipeline for the Dunder Business Entity
Resolution Challenge (Amazon ML Challenge 2026). The pipeline identifies which
records from Source 2 and Source 3 refer to the same real-world business entity
as each Source 1 record, scored by macro-averaged F₀.₅.

## Environment Setup

```bash
pip install -r requirements.txt
```

**Pinned dependencies**: pandas 2.2.x, numpy 2.x, scikit-learn 1.9.x,
lightgbm 4.7.x, rapidfuzz 3.x, duckdb 1.5.x, pyyaml 6.x, sparse-dot-topn, scipy.

---

## End-to-End Reproduction (from repo root)

All commands are run from the **repo root** (`-Dunder-Business-Entity-Resolution-Challenge/`).

### Option A: Complete end-to-end pipeline (Recommended)

```bash
python run_full_pipeline.py
```

This runs all steps end-to-end:
- Step 1: Normalization (reusing cached or generated)
- Step 2: 9-channel blocking (reusing CH1-CH5 if present, adding CH6-CH9)
- Step 3: Candidate deduplication & Phase 4 recall validation on Fold D
- Step 4: 15-feature pairwise extraction using RapidFuzz (chunked, memory-safe)
- Step 5: LightGBM training with early stopping & threshold tuning on MACRO F0.5
- Step 6: Isotonic calibration
- Step 7: Test set normalization & 9-channel blocking
- Step 8: `output/candidate_pairs.tsv` generation
- Step 9: Test set feature extraction & calibrated scoring
- Step 10: Decision resolution & `output/matching_results.tsv` generation
- Step 11: Automatic validation with `utils/validate_submission.py`
- Step 12: Automatic submission packaging into zip archive

---

## Output Files

| File | Description |
|------|-------------|
| `output/candidate_pairs.tsv` | Blocking output — all candidate entity pairs (one row per S1 entity) |
| `output/matching_results.tsv` | Final predicted matches — the leaderboard upload file |

---

## Validation

```bash
python utils/validate_submission.py \
    --matching  output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir  dataset/test
```

Exit code 0 = PASS.

Quick stats + validation:
```bash
python quick_validate.py
```

---

## Pipeline Architecture

```
9-Channel Blocking (Phase 4):
  CH1 exact_sorted              → DuckDB exact join on name_sorted
  CH2 exact_expanded_sorted     → DuckDB exact join on name_expanded_sorted
  CH3 token_rare (freq≤200)     → DuckDB inverted index, 1 shared rare token
  CH4 token_medium (freq≤5000)  → DuckDB inverted index, ≥2 shared medium tokens
  CH5 addr_composite            → DuckDB exact join: (4+digit num)+(4+char word)+country
  CH6 exact_stripped_sorted     → DuckDB exact join: leet-normalized + legal suffixes stripped
  CH7 name_prefix2              → DuckDB exact join: first 2 words + country (freq ≤ 50)
  CH8 addr_street               → DuckDB exact join: house number (1-6 digits) + street + country
  CH9 name_w1                   → DuckDB exact join: distinctive first word (len ≥ 6) + country
```

Feature Extraction (Phase 5): 15 pairwise features
  - RapidFuzz: token_sort_ratio, partial_ratio, QRatio, WRatio, expanded_sort_ratio
  - Exact: name_sorted_exact, name_alphanum_exact, name_prefix5_exact, addr_prefix4_match
  - Address: addr_token_jaccard, addr_numeric_match, addr_jaro_winkler
  - Country: country_match (open-set — France safe)
  - Structural: name_len_ratio, name_token_len_ratio

LightGBM (Phase 6):
  - num_leaves=127, learning_rate=0.03, n_estimators=1000
  - Trained on fold A (60%), threshold-tuned on fold B (10%)
  - scale_pos_weight handles class imbalance

Isotonic Calibration (Phase 7): fold C (15%), fold-disjoint from A and B

Decision Layer (Phase 8):
  - Apply F₀.₅-optimal threshold
  - Target-side dedup (greedy by calibrated probability)
  - Singletons: entities with no passing pairs → empty row (earns 1.0 F₀.₅)
```

---

## Project Structure

```
code/business_entity_resolution/
├── src/
│   ├── blocking.py            # Legacy entry-point
│   ├── matching.py            # Legacy entry-point
│   ├── blocking/
│   ├── features/
│   │   └── pairwise.py        # RapidFuzz + address feature extraction
│   ├── models/
│   │   └── matcher.py         # LightGBM binary classifier
│   ├── calibration/
│   │   └── calibrator.py      # Isotonic regression calibration
│   ├── decision/
│   │   └── resolver.py        # Threshold + dedup → matching_results.tsv
│   ├── pipeline/
│   │   ├── train_pipeline.py  # Phase 4–7 orchestrator (train)
│   │   └── run_inference.py   # Phase 15 orchestrator (test inference)
│   ├── data/
│   │   ├── eda.py             # Phase 1: EDA
│   │   └── splitter.py        # Phase 2: Entity-level split
│   └── evaluation/
│       └── scorer.py          # Macro F₀.₅ scorer
├── README.md                  # This file
└── requirements.txt           # Pinned dependencies

# Root-level pipeline scripts:
master_run.py            # Full end-to-end runner
add_new_channels.py      # Incremental: add CH6-CH9 to existing CH1-CH5
run_after_blocking.py    # Run Phases 5-16 after blocking is done
quick_validate.py        # Quick validation + stats
utils/validate_submission.py  # Official submission validator
```
