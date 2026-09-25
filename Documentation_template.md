# Documentation Template — Dunder Business Entity Resolution Challenge

## Team Name
<!-- Your team name here -->

## Team Members
<!-- List all members -->

---

## 1. Problem Understanding

Given noisy business records from three independent sources — S1 (deduplicated
reference), S2, and S3 — the task is to determine which S2/S3 records refer to
the same real-world business entity as each S1 record. Records share no common
identifiers. The output is scored by **macro-averaged F₀.₅** (precision
weighted 2× over recall) per S1 entity, including singletons.

Training data covers **US** and **India**. The test set adds an unseen third
country, **France**, which must be handled by country-agnostic logic.

---

## 2. Data Exploration (Phase 1)

**Record counts (train):**
- S1: 2,206,821 records
- S2: 5,034,616 records
- S3: 5,285,603 records

**Country split (S1 — reference set, train):**
- US: 1,323,633 (59.98%)
- India: 883,188 (40.02%)

**Match cardinality distribution (train GT, 2,206,821 S1 entities):**
- Singletons (0 matches): 123,247 (5.58%)
- 1-match: 119,157 (5.40%)
- Multi-match (>1 match): 1,964,417 (89.02%)
- Max matches for any single S1: 11
- Mean matches per S1 entity: 3.4613

**Target-side uniqueness:**
- S2: 3,693,619 IDs in GT — 0 duplicates → **unique=True**
- S3: 3,944,746 IDs in GT — 0 duplicates → **unique=True**
- Implication: each S2/S3 record belongs to at most one S1 entity

**Noise taxonomy (sampled from 25 known matching pairs):**
- Word-order swaps ("Eye Clinic LLC" → "LLC Eye Clinic")
- DBA prefix/suffix noise (`Smt`, `--`, `***`, `C0nstruction`, `Sri`)
- Legal suffix variants (Rd/Road, Ltd/Limited, Pvt/Private, Corp/Corporation)
- Punctuation differences (#, ., <NULL> literal in address fields)
- Transliteration in addresses (Indic script alongside Latin)
- Component reordering in address fields
- Typos: Sthoen→Stone, Ahute→Haute, Certhificatidons, Aelxandria, BEKELEY

**EDA artifact:** `artifacts/phase1_eda_summary.txt`

---

## 3. Blocking Strategy (Phase 4)

We use a **5-channel union** of deterministic blocking rules (no sklearn/scipy,
pure pandas). Each channel generates (S1_id, S2/S3_id) candidate pairs.

| Channel | Method | Key |
|---------|--------|-----|
| CH1 `exact_sorted` | Exact join | name_sorted (alphanum tokens sorted) |
| CH2 `exact_expanded_sorted` | Exact join | name_expanded_sorted (abbrev-expanded + sorted) |
| CH3 `token_rare` | Inverted index (1 rare token) | tokens with freq ≤ 150 in S2+S3 |
| CH4 `token_medium` | Inverted index (2 medium tokens) | tokens with freq ≤ 3,000, min 2 shared |
| CH5 `addr_composite` | Exact join | (4+-digit number) + (first street word) + country |

**Blocking recall (on training set, fold D held-out — 15% of S1 entities):**

| Split | Candidate pairs generated | True pairs recalled | Recall % |
|-------|--------------------------|--------------------|---------:|
| Fold D | _fill from artifacts/phase4_recall_report.txt_ | _ / _ | _% |

> Gate: recall ≥ 95% on fold D before advancing.

---

## 4. Matching / Similarity Model (Phases 5–6)

**Feature engineering (`src/features/pairwise.py`):**

| Feature | Type |
|---------|------|
| `name_token_sort_ratio` | RapidFuzz token_sort_ratio on `name_alphanum` |
| `name_partial_ratio` | RapidFuzz partial_ratio on `name_alphanum` |
| `name_expanded_sort_ratio` | RapidFuzz token_sort_ratio on `name_expanded` |
| `name_sorted_exact` | Binary: `name_sorted` views exactly equal |
| `name_alphanum_exact` | Binary: `name_alphanum` views exactly equal |
| `addr_token_jaccard` | Jaccard similarity of address alphanum token sets |
| `addr_numeric_match` | Binary: first numeric substring in address matches |
| `country_match` | Binary: `country` field string-equal |
| `name_len_ratio` | min/max of character lengths of `name_alphanum` |
| `name_token_len_ratio` | min/max of token counts of `name_alphanum` |

**Model:** LightGBM binary classifier  
**Training data:** Fold A (60% of S1 entities, ~1.32M entities)  
**Negatives:** blocker hard-negatives — candidates that are NOT true matches  
**Threshold tuning:** grid-search F₀.₅ on fold B (10%)  
**License:** LightGBM (MIT), RapidFuzz (MIT)

---

## 5. Calibration (Phase 7)

Isotonic regression (`sklearn.isotonic.IsotonicRegression`) fitted on fold C
(15% of S1 entities) — fold-disjoint from both training (fold A) and threshold
tuning (fold B). Reliability curve verified after fitting.

---

## 6. Post-processing / Decision Layer (Phase 8)

1. **Threshold application**: pairs above F₀.₅-optimised threshold → predicted match  
2. **Singleton protection**: if no pair passes threshold for an S1 entity, output empty row  
3. **Target-side deduplication**: if multiple S1 entities claim the same S2/S3 target,
   assign to the one with highest calibrated probability (greedy by score desc)

---

## 7. Results

| Split | Precision | Recall | F₀.₅ |
|-------|-----------|--------|------|
| Fold B (tune) | _fill_ | _fill_ | _fill_ |
| Fold D (val)  | _fill_ | _fill_ | _fill_ |
| Leaderboard   | _fill_ | _fill_ | _fill_ |

---

## 8. Challenges & Lessons Learned

- **Scale**: 1.73M test S1 entities require chunked processing to stay within 16 GB RAM.
- **France generalization**: all features and blocking are country-agnostic (open string label);
  no code path branches on `{US, India}` only.
- **Irreducible normalizer failures**: non-Latin scripts (Telugu, Devanagari) and OCR-level
  typos (Sthoen, C0nstruction) cannot be resolved by normalization alone — they rely on the
  edit-distance / fuzzy matching features in Phase 5.
- **F₀.₅ precision bias**: threshold was tuned specifically to maximize F₀.₅ (not accuracy
  or F1), which means we deliberately prefer higher precision and accept missing some recall.

---

## 9. References

- LightGBM (MIT): https://github.com/microsoft/LightGBM
- RapidFuzz (MIT): https://github.com/rapidfuzz/RapidFuzz
- scikit-learn IsotonicRegression: https://scikit-learn.org/stable/modules/isotonic.html
- Amazon ML Challenge 2026 Problem Statement: `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf`
