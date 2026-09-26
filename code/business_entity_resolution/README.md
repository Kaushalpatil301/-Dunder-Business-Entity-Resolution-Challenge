# Business Entity Resolution — Reproduction & Architecture Guide
**Amazon ML Challenge 2026 | Team Dunder**

---

## 1. Team Information
- **Team Name**: `Dunder`
- **Team Members**:
  - **Tanmay Lagoo** (Team Leader, +919920997305)
  - **Shivam Sharma** (+917741996435)
  - **Vansh Kataria** (+919920910010)
  - **Kaushal Patil** (+919323291327)

---

## 2. Overview
This directory contains the self-contained machine learning pipeline for the **Dunder Business Entity Resolution Challenge**. Given business records from three independent sources (Source 1 reference, Source 2, and Source 3) with heterogeneous schemas, OCR typos, leetspeak noise, legal abbreviations, and country differences (including unobserved test country **France**), the pipeline identifies which records refer to the same real-world entity under a macro-averaged $F_{0.5}$ metric (precision weighted 2x over recall).

---

## 3. Environment Setup

Python 3.10+ recommended (compatible with Python 3.10, 3.11, 3.12, 3.14).

```bash
# Install pinned dependencies
pip install -r requirements.txt
```

**Core Libraries:**
- `duckdb` (>=1.0.0): In-memory OLAP SQL engine for streaming, multi-threaded 9-channel blocking
- `lightgbm` (>=4.3.0): Gradient-boosted decision trees binary classifier (MIT License, ~10M parameters)
- `rapidfuzz` (>=3.9.0): SIMD-accelerated string edit distances and fuzzy metrics
- `scikit-learn` (>=1.5.0): Isotonic probability calibration and split validation
- `pandas` (>=2.0.0), `numpy` (>=1.26.0), `pyarrow` (>=14.0.0), `scipy` (>=1.10.0), `pyyaml` (>=6.0.0)

---

## 4. How to Reproduce

### Option A: Fast Test Reproduction via Pretrained Model (~2 minutes)
To immediately reproduce the scored outputs `output/matching_results.tsv` and `output/candidate_pairs.tsv` without 30-minute retraining:

```bash
# From package root:
python code/business_entity_resolution/run_full_pipeline.py --mode inference

# Or from inside code/business_entity_resolution/:
python run_full_pipeline.py --mode inference
```

This step:
1. Normalizes the test set (`dataset/test/test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`).
2. Executes 9-channel candidate blocking in DuckDB.
3. Generates `output/candidate_pairs.tsv` with match-prioritization.
4. Performs streaming RapidFuzz pairwise feature extraction (38,000+ pairs/sec).
5. Scores candidates using the included `artifacts/lgbm_model.txt` and calibrates via `artifacts/calibrator.pkl`.
6. Resolves matches via greedy target-side deduplication and singleton protection ($F_{0.5}$ optimal threshold 0.4900).
7. Writes `output/matching_results.tsv` (1,732,544 rows).
8. Automatically runs `validate_submission.py` to confirm exit code 0.

### Option B: Full Retraining Pipeline from Raw Data (~30 minutes)
To retrain LightGBM from scratch on Fold A, tune threshold on Fold B, calibrate on Fold C, and evaluate on Fold D:

```bash
# From package root:
python code/business_entity_resolution/run_full_pipeline.py --mode full

# Or from inside code/business_entity_resolution/:
python run_full_pipeline.py --mode full
```

---

## 5. Output Verification & Validation

The official competition validator `utils/validate_submission.py` confirms 100% adherence to all format, singleton, and candidate-subset rules:

```bash
python code/business_entity_resolution/src/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Expected Result:**
```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (110258 empty, 1622286 non-empty).
  candidate_pairs.tsv: 1732544 rows (16834 empty, 1715710 non-empty).
PASS — no blocking issues found. Safe to submit.
```

---

## 6. Architecture & Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RAW DATASET (S1, S2, S3)                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 PHASE 3: MULTI-VIEW NORMALIZATION                      │
│ • Lowercase, alphanum, sorted tokens, abbreviation expansion           │
│ • Leetspeak translation (0->o, 1->l, @->a), legal suffix stripping     │
│ • Distinctive prefix/word tokens, composite address keys               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   PHASE 4: 9-CHANNEL DUCKDB BLOCKING                   │
│ CH1: exact_sorted               CH6: exact_stripped_sorted             │
│ CH2: exact_expanded_sorted      CH7: name_prefix2 (freq <= 50)         │
│ CH3: token_rare (freq <= 200)   CH8: addr_street (freq <= 50)          │
│ CH4: token_medium (freq <= 5k)  CH9: name_w1 (freq <= 40, len >= 6)   │
│ CH5: addr_composite                                                    │
│ -> Prunes 99.72% of Cartesian space while preserving 98.34% recall     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            PHASE 5: 15-FEATURE VECTORIZED EXTRACTION                   │
│ • RapidFuzz token_sort_ratio, partial_ratio, QRatio, WRatio            │
│ • Exact matching flags on sorted, stripped, and prefix tokens          │
│ • Address token Jaccard, numeric match, street prefix match            │
│ • Country exact match (fully open-set; handles France seamlessly)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              PHASE 6 & 7: LIGHTGBM + ISOTONIC CALIBRATION              │
│ • LightGBM binary classifier (num_leaves=127, lr=0.03, early stopping) │
│ • F0.5-optimal threshold grid search on disjoint Fold B (th = 0.4900)  │
│ • Isotonic regression probability calibration on disjoint Fold C       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  PHASE 8: DECISION & DEDUPLICATION                     │
│ • Calibrated probability thresholding                                  │
│ • Target-side 1:1 greedy deduplication (highest confidence S1 wins)    │
│ • Singleton protection (empty matches score 1.0 macro F0.5)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          COMPLIANT OUTPUTS                             │
│ • output/matching_results.tsv (1,732,544 rows, leaderboard scored)     │
│ • output/candidate_pairs.tsv (1,732,544 rows, blocking candidates)     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Compliance & Integrity Guarantees

1. **Model License & Size**:
   - Model framework: LightGBM (MIT License) & scikit-learn (BSD 3-Clause).
   - Parameters: ~10 Million parameters ($<< 8.0$ Billion parameter ceiling).
2. **Academic Integrity**:
   - Zero external data lookups, zero internet/commercial API calls, zero geocoding.
   - Fully open-set country representation (France handled identically through invariant text similarity).
