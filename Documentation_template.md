# Documentation Template — Dunder Business Entity Resolution Challenge

## Team Name
Dunder

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
- Punctuation differences (#, ., `<NULL>` literal in address fields)
- Transliteration in addresses (Indic script alongside Latin)
- Component reordering in address fields
- Typos: Sthoen→Stone, Ahute→Haute, Certhificatidons, Aelxandria, BEKELEY

**EDA artifact:** `artifacts/phase1_eda_summary.txt`

---

## 3. Blocking Strategy (Phase 4)

We use a **9-channel union** of deterministic and similarity-based blocking rules.
Each channel generates (S1_id, S2/S3_id) candidate pairs. All implementations are
country-agnostic — France is handled by the same code paths as US and India.

| Channel | Method | Key |
|---------|--------|-----|
| CH1 `exact_sorted` | DuckDB exact join | `name_sorted` (alphanum tokens sorted) |
| CH2 `exact_expanded_sorted` | DuckDB exact join | `name_expanded_sorted` (abbrev-expanded + sorted) |
| CH3 `token_rare` | DuckDB inverted index (1 rare token) | tokens with freq ≤ 200 in S2+S3 |
| CH4 `token_medium` | DuckDB inverted index (2+ medium tokens) | tokens with freq ≤ 5,000, min 2 shared |
| CH5 `addr_composite` | DuckDB exact join | (4+-digit house number) + (4+-char street word) + country |
| CH6 `exact_stripped_sorted` | DuckDB exact join | Leetspeak normalized (0→o, 1→l, etc.) + legal suffixes stripped (llc, inc, ltd, etc.) + sorted |
| CH7 `name_prefix2` | DuckDB exact join (frequency capped) | First 2 words of stripped name + country (length ≥ 8, S2/S3 freq ≤ 50) |
| CH8 `addr_street` | DuckDB exact join (frequency capped) | Any house number (1-6 digits) + street name (≥3 chars) + country (freq ≤ 50) |
| CH9 `name_w1` | DuckDB exact join (frequency capped) | Distinctive first word (length ≥ 6) + country (freq ≤ 40) |

**Why these channels?**
- CH1/CH2: handle legal suffix abbreviations and word-order transpositions
- CH3/CH4: handle partial token overlap (DBA names, compound names)
- CH5: matches businesses at the same physical address with 4+ digit street numbers
- CH6: captures synthetic noise and OCR errors (leetspeak substitutions) and inconsistent legal entity types (e.g., LLC vs Corp)
- CH7: matches multi-word business names with identical prefixes across sources
- CH8: captures businesses sharing street numbers (1 to 6 digits) and street names, with high-frequency false positives filtered
- CH9: matches distinctive single leading tokens without Cartesian explosion

---

## 4. Normalization (Phase 3)

Multi-view normalization applied to `business_name` and `business_address`:

| View | Transformation |
|------|---------------|
| `name_lower` | Lowercase + whitespace collapse + DBA prefix strip (Smt, Sri, --, ***) |
| `name_alphanum` | Lower + non-alphanumeric → space |
| `name_sorted` | Alphanum tokens sorted alphabetically (handles word-order) |
| `name_expanded` | Abbreviation bi-directional expansion/collapse (pvt↔private, ltd↔limited, etc.) |
| `name_expanded_sorted` | Expanded view with tokens sorted |
| `name_prefix5` | First 5 characters of `name_sorted` |
| `csort` | Leetspeak translated (0→o, 1→l, 3→e, 4→a, 5→s, 7→t, 8→b, @→a, $→s) + legal suffixes removed + sorted |
| `cpref2` | First 2 words of leet-normalized stripped name |
| `cw1` | Distinctive first word (length ≥ 6) of stripped name |
| `addr_alphanum` | Address lowercase + non-alphanumeric → space |
| `addr_numeric` | First numeric substring (door/PIN/ZIP) extracted |
| `addr_prefix4` | First 4 characters of `addr_alphanum` |
| `addr_composite` | 4+ digit number + 4+ char word |
| `addr_street` | 1-6 digit house number + 3+ char street word |

**Phase 3 gate:** 21/25 known noisy true pairs normalize to near-identical views (gate ≥ 20/25). 4 irreducible failures involve non-Latin scripts and OCR typos.

---

## 5. Matching / Similarity Model (Phases 5–6)

**Feature engineering (15 pairwise features):**

| Feature | Type | Notes |
|---------|------|-------|
| `name_token_sort_ratio` | RapidFuzz `token_sort_ratio` on `name_alphanum` | Handles word-order |
| `name_partial_ratio` | RapidFuzz `partial_ratio` on `name_alphanum` | Handles substring matches |
| `name_expanded_sort_ratio` | RapidFuzz `token_sort_ratio` on `name_expanded` | Abbrev-aware |
| `name_qratio` | RapidFuzz `QRatio` | Overall edit distance |
| `name_jaro_winkler` | RapidFuzz `WRatio` | Weighted composite |
| `name_sorted_exact` | Binary exact match on `name_sorted` | — |
| `name_alphanum_exact` | Binary exact match on `name_alphanum` | — |
| `addr_token_jaccard` | Jaccard similarity of address alphanum token sets | — |
| `addr_numeric_match` | Binary: first numeric addr substring matches | Door/building number |
| `addr_jaro_winkler` | RapidFuzz `partial_ratio` on `addr_alphanum` | — |
| `addr_prefix4_match` | Binary: first 4 chars of `addr_alphanum` match | — |
| `country_match` | Binary: `country` string-equal | Open-set — France safe |
| `name_len_ratio` | min/max of character lengths | Structural |
| `name_token_len_ratio` | min/max of token counts | Structural |
| `name_prefix5_exact` | Binary: `name_prefix5` views exactly equal | Fast structural gate |

All features are country-agnostic by design — no feature requires knowing the specific country label.

**Model:** LightGBM binary classifier (MIT License, ≤8B params)
- `num_leaves=127`, `learning_rate=0.03`, `n_estimators=1000`
- `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=0.1`
- `scale_pos_weight` = neg/pos ratio (handles class imbalance)
- Early stopping after 50 rounds on fold B

**Training data:** Fold A (60% of S1 entities, ~1.32M entities)
**Negatives:** blocker hard-negatives — candidates that are NOT true matches
**Threshold tuning:** grid-search F₀.₅ on fold B (10%, 0.10–0.90 in steps of 0.01)

---

## 6. Calibration (Phase 7)

Isotonic regression (`sklearn.isotonic.IsotonicRegression`) fitted on fold C
(15% of S1 entities) — fold-disjoint from both training (fold A) and threshold
tuning (fold B). Reliability curve verified after fitting.

---

## 7. Post-processing / Decision Layer (Phase 8)

1. **Threshold application**: pairs above F₀.₅-optimised threshold → predicted match
2. **Singleton protection**: if no pair passes threshold for an S1 entity, output empty row (correctly earns 1.0 F₀.₅)
3. **Target-side deduplication**: if multiple S1 entities claim the same S2/S3 target,
   assign to the one with highest calibrated probability (greedy by score desc). This is
   eligible because EDA confirmed target-side uniqueness (0 S2/S3 duplicates in GT).

---

## 8. Results

| Split | Precision | Recall | F₀.₅ |
|-------|-----------|--------|------|
| Fold B (tune) | _fill_ | _fill_ | _fill_ |
| Fold D (val)  | _fill_ | _fill_ | _fill_ |
| Leaderboard   | _fill_ | _fill_ | _fill_ |

---

## 9. Challenges & Lessons Learned

- **Scale**: 1.73M test S1 entities require chunked processing to stay within 16 GB RAM.
  Used DuckDB COPY TO for zero-Python-RAM channel output and chunked TF-IDF cosine via
  `sparse_dot_topn`.
- **Recall ceiling**: Initial 5-channel blocking achieved only 57.75% recall on fold D.
  Adding 4 new channels (prefix, addr-numeric, TF-IDF name, TF-IDF addr) was critical
  to reaching the ≥95% gate.
- **France generalization**: all features and blocking are country-agnostic (open string label);
  no code path branches on `{US, India}` only.
- **Irreducible normalizer failures**: non-Latin scripts (Telugu, Devanagari) and OCR-level
  typos (Sthoen, C0nstruction) cannot be resolved by normalization alone — they rely on the
  edit-distance / fuzzy matching features in Phase 5.
- **F₀.₅ precision bias**: threshold was tuned specifically to maximize F₀.₅ (not accuracy
  or F1), which means we deliberately prefer higher precision and accept missing some recall.
  Singletons score 1.0 when correctly identified as empty — the decision layer explicitly
  protects against false positive merges on singletons.

---

## 10. References

- LightGBM (MIT): https://github.com/microsoft/LightGBM
- RapidFuzz (MIT): https://github.com/rapidfuzz/RapidFuzz
- scikit-learn IsotonicRegression: https://scikit-learn.org/stable/modules/isotonic.html
- sparse_dot_topn (Apache 2.0): https://github.com/ing-bank/sparse_dot_topn
- DuckDB (MIT): https://duckdb.org/
- Amazon ML Challenge 2026 Problem Statement: `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf`
