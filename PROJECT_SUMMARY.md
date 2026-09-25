# Dunder Business Entity Resolution — Project Summary

## Overview
Complete end-to-end ML pipeline for the Amazon ML Challenge 2026: Business Entity Resolution. Given noisy business records from 3 independent sources (S1 = deduplicated reference, S2, S3), the system determines which S2/S3 records refer to the same real-world entity as each S1 record. Evaluated on macro-averaged F₀.₅ (precision weighted 2× over recall).

---

## Repository Structure

```
.
├── config/
│   ├── config.py           # YAML config loader (single source of truth)
│   └── pipeline.yaml       # All hyperparameters, paths, split ratios
├── src/ (in code/business_entity_resolution/src/)
│   ├── normalize/          # Phase 3: Multi-view normalization
│   ├── blocking/           # Phase 4: 9-channel candidate generation
│   ├── features/           # Phase 5: Pairwise similarity features
│   ├── models/             # Phase 6: LightGBM classifier
│   ├── calibration/        # Phase 7: Isotonic regression
│   ├── decision/           # Phase 8: Threshold + deduplication
│   ├── data/               # EDA, entity splitting
│   ├── evaluation/         # Macro F0.5 scorer
│   └── pipeline/           # Orchestration scripts
├── run_full_pipeline.py    # Master script (1200+ lines, self-contained)
├── package_submission.py   # Creates submission zip
├── utils/validate_submission.py  # Official validator
├── artifacts/              # Cached intermediate outputs
├── output/                 # Final matching_results.tsv, candidate_pairs.tsv
├── dataset/train/          # Training TSVs + ground truth
├── dataset/test/           # Test TSVs (no labels)
└── Documentation_template.md  # Filled methodology document
```

---

## Phase-by-Phase Implementation

### Phase 1: Exploratory Data Analysis (`src/data/eda.py`)
- **Record counts (train)**: S1=2,206,821 | S2=5,034,616 | S3=5,285,603
- **Country split (S1)**: US=1,323,633 (59.98%) | India=883,188 (40.02%)
- **Match cardinality**: Singletons=5.58%, 1-match=5.40%, Multi-match=89.02% (max 11, mean 3.46)
- **Target uniqueness**: S2/S3 entities in GT have **zero duplicates** — each belongs to at most one S1
- **Noise taxonomy**: Word-order swaps, DBA prefixes (Smt, Sri, --, ***), legal suffix variants (Rd/Road, Ltd/Limited, Pvt/Private, Corp/Corporation), punctuation, transliteration, component reordering, OCR typos (Sthoen→Stone, C0nstruction)

### Phase 2: Entity-Level Split (`src/data/splitter.py`)
- Stratified 4-fold split by `is_singleton` to preserve singleton ratio
- Fold A: 60% (training), Fold B: 10% (threshold tuning), Fold C: 15% (calibration), Fold D: 15% (recall gate)
- Saved to `artifacts/entity_split.tsv`

### Phase 3: Multi-View Normalization (`src/normalize/normalizer.py`)
Produces 5 views per field (`business_name`, `business_address`):

| View | Transformation |
|------|---------------|
| `_lower` | Lowercase + whitespace collapse + DBA prefix strip (Smt, Sri, --, ***) |
| `_alphanum` | Non-alphanumeric → space |
| `_sorted` | Alphanum tokens sorted alphabetically (handles word-order noise) |
| `_expanded` | Bi-directional abbreviation expansion (pvt↔private, ltd↔limited, inc↔incorporated, corp↔corporation, rd↔road, st↔street, etc.) + US state abbreviations |
| `_raw` | Original preserved |

**Unicode handling**: NFD accent stripping (Á→A) while preserving non-Latin scripts (Indic, CJK) intact.
**Phase 3 gate**: 21/25 known true pairs normalize to near-identical `sorted`/`expanded` views.

### Phase 4: 9-Channel Blocking (`src/blocking/` + `run_full_pipeline.py`)
All channels use DuckDB exact joins with frequency caps to control candidate explosion. Country-agnostic by design.

| Channel | Method | Key | Frequency Cap |
|---------|--------|-----|---------------|
| CH1 `exact_sorted` | Exact join | `name_sorted` | — |
| CH2 `exact_expanded_sorted` | Exact join | `name_expanded_sorted` | — |
| CH3 `token_rare` | Inverted index (1 shared) | Rare tokens (freq ≤ 200 in S2+S3) | — |
| CH4 `token_medium` | Inverted index (2+ shared) | Medium tokens (freq ≤ 5,000) | min 2 shared |
| CH5 `addr_composite` | Exact join | 4+ digit number + 4+ char word + country | ≤ 50 |
| CH6 `exact_stripped_sorted` | Exact join | Leetspeak (0→o,1→l,3→e,4→a,5→s,7→t,8→b,@→a,$→s) + legal suffixes stripped + sorted | ≤ 100 |
| CH7 `name_prefix2` | Exact join | First 2 words of stripped name + country (len ≥ 8) | ≤ 50 |
| CH8 `addr_street` | Exact join | 1-6 digit house number + 3+ char street word + country | ≤ 50 |
| CH9 `name_w1` | Exact join | First word (len ≥ 6) + country | ≤ 40 |

**Deduplication**: Union of all channels → single flat candidate set (`candidate_pairs_all_folds_flat.tsv`, ~100M+ pairs)

**Recall verification**: Phase 4 recall on Fold D measured via SQL — gate ≥95% required before proceeding.

### Phase 5: Feature Extraction (`src/features/pairwise.py` + `run_full_pipeline.py`)
15 pairwise features computed via vectorized RapidFuzz + DuckDB streaming joins:

| Feature | Type | Source |
|---------|------|--------|
| `name_token_sort_ratio` | RapidFuzz token_sort_ratio | `name_alphanum` |
| `name_partial_ratio` | RapidFuzz partial_ratio | `name_alphanum` |
| `name_expanded_sort_ratio` | RapidFuzz token_sort_ratio | `name_expanded` |
| `name_qratio` | RapidFuzz QRatio | `name_alphanum` |
| `name_jaro_winkler` | RapidFuzz JaroWinkler | `name_alphanum` |
| `name_sorted_exact` | Binary | `name_sorted` equality |
| `name_alphanum_exact` | Binary | `name_alphanum` equality |
| `addr_token_jaccard` | Jaccard | `addr_alphanum` token sets |
| `addr_numeric_match` | Binary | First numeric substring match (len ≥ 2) |
| `addr_jaro_winkler` | RapidFuzz partial_ratio | `addr_alphanum` |
| `addr_prefix4_match` | Binary | First 4 chars of `addr_alphanum` |
| `country_match` | Binary | Country string equality (open-set) |
| `name_len_ratio` | min/max char length | Structural |
| `name_token_len_ratio` | min/max token count | Structural |
| `name_prefix5_exact` | Binary | `name_prefix5` equality |

**Training data construction**:
- Fold A: ~700K true positives + ~1.2M blocker negatives (sampled 3%)
- Fold B: All true positives + ~500K blocker negatives (sampled 4%)
- Fold C: ~100K true positives + ~100K blocker negatives (sampled 1%)
- Saved as Parquet: `features_fold_a/b/c.parquet`

### Phase 6: LightGBM Training (`src/models/matcher.py` + `run_full_pipeline.py`)
- **Model**: LGBMClassifier (MIT license, <8B params)
- **Params**: `num_leaves=127`, `learning_rate=0.05`, `n_estimators=600`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=0.1`, `scale_pos_weight = neg/pos`
- **Early stopping**: 30 rounds on Fold B validation
- **Threshold tuning**: Grid search 0.15–0.85 step 0.02 maximizing **macro F₀.₅ on Fold B** (including singletons)
- **Output**: `artifacts/lgbm_model.txt`, `artifacts/threshold.txt`

### Phase 7: Calibration (`src/calibration/calibrator.py`)
- IsotonicRegression (sklearn) fitted on Fold C (disjoint from train A and tune B)
- `out_of_bounds="clip"`, `y_min=0.0`, `y_max=1.0`
- Output: `artifacts/calibrator.pkl`

### Phase 8: Decision Layer (`src/decision/resolver.py` + `run_full_pipeline.py`)
1. **Threshold application**: Calibrated probability ≥ tuned threshold → predicted match
2. **Singleton protection**: No passing pairs → empty prediction (earns 1.0 F₀.₅)
3. **Target-side greedy deduplication**: If multiple S1 claim same S2/S3 target, assign to highest calibrated probability (valid because EDA confirmed target uniqueness)

### Phase 9: Test Inference (`run_full_pipeline.py`)
- Multi-view normalization on test sources (includes unseen France)
- Same 9-channel blocking → `test_candidates_flat.tsv`
- **Zero-disk streaming scoring**: DuckDB joins → Arrow batches → RapidFuzz features → LightGBM → Isotonic → threshold filter
- Batch size 200K pairs, 10 threads for inference
- Output: `passing_pairs` DataFrame (s1_id, target_id, cal_prob)

### Phase 10: Output Generation (`run_full_pipeline.py`)
- **matching_results.tsv**: One row per test S1 entity, comma-separated matched S2/S3 IDs (empty for singletons)
- **candidate_pairs.tsv**: Top 95 candidates per S1 from blocking + all matched IDs prioritized, max 100 per row
- Both validated via `utils/validate_submission.py` (checks format, completeness, no duplicates, valid IDs)

### Phase 11: Submission Packaging (`package_submission.py`)
Creates `<team>_submission_<timestamp>.zip` with:
```
output/matching_results.tsv
output/candidate_pairs.tsv
code/business_entity_resolution/src/ (all .py)
code/business_entity_resolution/README.md
code/business_entity_resolution/requirements.txt
Documentation_template.md
config/pipeline.yaml, config.py
utils/validate_submission.py
run_full_pipeline.py, master_run.py, add_new_channels.py, run_after_blocking.py, quick_validate.py
```

---

## Key Technical Decisions

1. **DuckDB for scale**: SQL joins on 100M+ candidate pairs without loading into Python memory; 12-14GB memory limits
2. **Macro F₀.₅ optimization**: Threshold tuned on per-entity macro average (not micro), including singletons
3. **Disjoint data splits**: Train (A) / Tune (B) / Calibrate (C) / Recall-gate (D) — no leakage
4. **Country-agnostic throughout**: No hardcoded {US, India} logic; France handled identically
5. **Target uniqueness exploited**: Greedy deduplication valid because 0 duplicates in GT
6. **Artifact caching**: Every phase checks for existing outputs — re-runs only what changed
7. **Vectorized normalization**: Pandas str ops + single `.apply` for token sort; no row-wise loops on large frames

---

## Dependencies (pinned in requirements.txt)
- pandas, numpy, pyarrow
- duckdb ≥0.9
- lightgbm ≥4.0
- rapidfuzz ≥3.0
- scikit-learn ≥1.3
- scipy, pyyaml

---

## How to Run

```bash
# Full end-to-end (trains if no cached model, else reuses)
python run_full_pipeline.py

# Or use master_run.py for step-by-step
python master_run.py

# Validate outputs before submission
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test

# Create submission zip
python package_submission.py --team-name Dunder
```

---

## Current Status
- ✅ All pipeline code complete and modular
- ✅ Documentation template fully filled (phases 1-10)
- ✅ Validator integrated
- ✅ Packaging script ready
- ⚠️ **Output files currently empty** — need to execute `run_full_pipeline.py` to generate predictions
- ⚠️ **Results table in docs unfilled** — add F₀.₅ scores after run

---

## Expected Performance
Based on architecture (9-channel blocking ≥95% recall gate, LightGBM with macro F₀.₅ tuning, isotonic calibration, target deduplication), the system is designed to achieve **competitive F₀.₅ on both public and private leaderboards**. The precision-heavy tuning and singleton protection directly optimize the competition metric.