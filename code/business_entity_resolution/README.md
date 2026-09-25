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

**Pinned dependencies**: pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.0,
lightgbm 4.3.0, rapidfuzz 3.9.3, pyyaml 6.0.1.

## End-to-End Reproduction

All commands are run from the **repo root** (`-Dunder-Business-Entity-Resolution-Challenge/`).

---

### Step 1 — Prepare Data (no action needed)

Dataset is already structured under `dataset/train/` and `dataset/test/`.

---

### Step 2 — Verify Entity Split Exists

The entity split (fold A/B/C/D) was generated in Phase 2 and stored at
`artifacts/entity_split.tsv`. If missing, regenerate it:

```bash
python code/business_entity_resolution/src/data/splitter.py
```

---

### Step 3 — Train Pipeline (Phases 4–7)

Runs blocking, feature extraction, LightGBM training, and isotonic
calibration in sequence. Artifacts are saved under `artifacts/`.

```bash
python code/business_entity_resolution/src/pipeline/train_pipeline.py
```

This produces:
- `artifacts/lgbm_model.txt` — trained model
- `artifacts/threshold.txt` — F₀.₅-optimised threshold
- `artifacts/calibrator.pkl` — isotonic calibrator
- `artifacts/phase4_recall_report.txt` — Phase 4 gate evidence

---

### Step 4 — Generate Candidates for Test Set

```bash
python code/business_entity_resolution/src/blocking.py \
    --input_dir ../../dataset/test \
    --output    ../../output/candidate_pairs.tsv
```

*(Or equivalently, from within `code/business_entity_resolution/`:)*

```bash
python src/blocking.py \
    --input_dir ../../dataset/test \
    --output    ../../output/candidate_pairs.tsv
```

---

### Step 5 — Run Matching (Phases 5–8)

```bash
python src/matching.py \
    --candidates ../../output/candidate_pairs.tsv \
    --input_dir  ../../dataset/test \
    --output     ../../output/matching_results.tsv
```

---

### Step 6 — Validate Submission

```bash
python ../../utils/validate_submission.py \
    --matching  ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir  ../../dataset/test
```

Exit code 0 = PASS. Only then proceed to packaging.

---

### Alternative: Full End-to-End Inference (Test Only)

If artifacts are already trained, you can run everything in one call:

```bash
python code/business_entity_resolution/src/pipeline/run_inference.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `output/candidate_pairs.tsv` | Blocking output — all candidate entity pairs (one row per S1 entity) |
| `output/matching_results.tsv` | Final predicted matches — the leaderboard upload file |

## Project Structure

```
code/business_entity_resolution/
├── src/
│   ├── blocking.py            # Entry-point: blocking for any split
│   ├── matching.py            # Entry-point: feature+model+decision
│   ├── blocking/
│   │   ├── fast_normalizer.py # Vectorized normalization for blocking
│   │   └── union.py           # 5-channel blocking union orchestrator
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
```
