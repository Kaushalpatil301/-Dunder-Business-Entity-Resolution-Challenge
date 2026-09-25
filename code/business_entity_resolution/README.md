# Business Entity Resolution — Reproduction Guide

## Overview
This folder contains the complete pipeline for the Dunder Business Entity Resolution Challenge.

## Environment Setup

```bash
pip install -r requirements.txt
```

## How to Reproduce End-to-End

### Step 1 — Prepare Data
Place the raw dataset files under `../../dataset/train/` and `../../dataset/test/` (already structured in the repo root).

### Step 2 — Blocking
Generate the candidate pairs that reduce the comparison space:

```bash
python src/blocking.py \
    --input_dir ../../dataset \
    --output ../../output/candidate_pairs.tsv
```

### Step 3 — Matching
Run the matching model on the candidate pairs:

```bash
python src/matching.py \
    --candidates ../../output/candidate_pairs.tsv \
    --input_dir ../../dataset \
    --output ../../output/matching_results.tsv
```

### Step 4 — Validate
```bash
python ../../utils/validate_submission.py \
    --submission ../../output/matching_results.tsv
```

## Output Files
| File | Description |
|------|-------------|
| `output/candidate_pairs.tsv` | Blocking output — all candidate entity pairs |
| `output/matching_results.tsv` | Final predicted matches (leaderboard upload file) |

## Project Structure
```
code/business_entity_resolution/
├── src/          # all source code
├── README.md     # this file
└── requirements.txt
```
