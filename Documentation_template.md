# Documentation Template — Dunder Business Entity Resolution Challenge

## Team Name
Dunder

## Team Members
- Tanmay Lagoo (Team Leader, +919920997305)
- Shivam Sharma (+917741996435)
- Vansh Kataria (+919920910010)
- Kaushal Patil (+919323291327)

---

## 1. Problem Understanding

Given noisy business records from three independent sources — S1 (deduplicated
reference), S2, and S3 — the task is to determine which S2/S3 records refer to
the same real-world business entity as each S1 record. Records share no common
identifiers. The output is scored by **macro-averaged F_0.5** (precision
weighted 2x over recall) per S1 entity, including singletons.

**F_0.5 formula:**
```
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
```

**Key design considerations driven by F_0.5:**
- Precision is weighted 2x over recall — false merges (matching two different businesses) are penalised more than missed links.
- Singletons (S1 entities with no true matches) score 1.0 when correctly predicted as empty, and 0.0 when any match is predicted. This means correctly identifying singletons earns full credit, and false merges on them are maximally penalised.
- Our entire pipeline — from blocking through to the decision layer — is designed around this precision-heavy objective.

Training data covers **US** and **India**. The test set additionally contains a third country, **France**, that does **not** appear in training. Our pipeline treats `country` as an open-set string label: no feature, blocking channel, or code path hard-codes or one-hots `{US, India}`. France entities are handled identically through country-agnostic text similarity.

---

## 2. Data Exploration

**Record counts (train):**
- S1: 2,206,821 records (deduplicated reference source)
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
- S2: 3,693,619 IDs in GT — 0 duplicates (unique=True)
- S3: 3,944,746 IDs in GT — 0 duplicates (unique=True)
- Implication: each S2/S3 record belongs to at most one S1 entity. This enables greedy target-side deduplication in the decision layer.

**Noise taxonomy (sampled from 25 known matching pairs):**
- Word-order swaps ("Eye Clinic LLC" -> "LLC Eye Clinic")
- DBA prefix/suffix noise (Smt, --, ***, C0nstruction, Sri)
- Legal suffix variants (Rd/Road, Ltd/Limited, Pvt/Private, Corp/Corporation)
- Punctuation differences (#, ., &/and, `<NULL>` literal in address fields)
- Transliteration in addresses (Indic script alongside Latin)
- Component reordering in address fields
- Typos: Sthoen->Stone, Ahute->Haute, Certhificatidons, Aelxandria, BEKELEY

---

## 3. Blocking Strategy

We use a **9-channel union** of deterministic blocking rules, all executed via DuckDB
in-memory SQL joins for scalability. Each channel generates (S1_id, S2/S3_id) candidate
pairs. The union of all channels forms the candidate set that is fed to the matching model.
All implementations are country-agnostic — France is handled by the same code paths as US and India.

| Channel | Method | Blocking Key |
|---------|--------|--------------|
| CH1 `exact_sorted` | DuckDB exact join | `name_sorted` (alphanum tokens sorted alphabetically) |
| CH2 `exact_expanded_sorted` | DuckDB exact join | `name_expanded_sorted` (abbreviation-expanded + sorted) |
| CH3 `token_rare` | DuckDB inverted index | Any 1 shared rare token (token freq <= 200 in S2+S3) |
| CH4 `token_medium` | DuckDB inverted index | >= 2 shared medium-freq tokens (token freq <= 5,000) |
| CH5 `addr_composite` | DuckDB exact join | (4+ digit house number) + (4+ char street word) + country |
| CH6 `exact_stripped_sorted` | DuckDB exact join | Leetspeak-normalised (0->o, 1->l, etc.) + legal suffixes stripped + sorted |
| CH7 `name_prefix2` | DuckDB exact join (freq-capped) | First 2 words of stripped name + country (S2/S3 freq <= 50) |
| CH8 `addr_street` | DuckDB exact join (freq-capped) | House number (1-6 digits) + street name (>= 3 chars) + country (freq <= 50) |
| CH9 `name_w1` | DuckDB exact join (freq-capped) | Distinctive first word (len >= 6) + country (freq <= 40) |

**Rationale for channel selection:**
- CH1/CH2: capture exact name matches after normalising word order and legal suffixes (Corp->Corporation, Pvt->Private, etc.)
- CH3/CH4: handle partial token overlap for DBA names and compound business names
- CH5: matches businesses at the same physical address using 4+ digit street numbers
- CH6: captures synthetic noise/OCR errors (leetspeak: C0nstruction->Construction) and inconsistent legal entity types
- CH7: matches multi-word business names sharing identical leading prefixes across sources
- CH8: captures businesses sharing street numbers and street names with high-frequency false positives filtered
- CH9: matches distinctive single leading tokens (length >= 6) without Cartesian explosion via frequency caps

**Multi-view normalization** applied to `business_name` and `business_address` before blocking:

| View | Transformation |
|------|---------------|
| `name_lower` | Lowercase + whitespace collapse + DBA prefix strip (Smt, Sri, --, ***) |
| `name_alphanum` | Lower + non-alphanumeric -> space |
| `name_sorted` | Alphanum tokens sorted alphabetically (handles word-order) |
| `name_expanded` | Abbreviation bidirectional expansion/collapse (pvt<->private, ltd<->limited, etc.) |
| `name_expanded_sorted` | Expanded view with tokens sorted |
| `name_prefix5` | First 5 characters of `name_sorted` |
| `csort` | Leetspeak translated + legal suffixes removed + sorted |
| `cpref2` | First 2 words of leet-normalised stripped name |
| `cw1` | Distinctive first word (length >= 6) of stripped name |
| `addr_alphanum` | Address lowercase + non-alphanumeric -> space |
| `addr_numeric` | First numeric substring (door/PIN/ZIP) extracted |
| `addr_prefix4` | First 4 characters of `addr_alphanum` |
| `addr_composite` | 4+ digit number + 4+ char word |
| `addr_street` | 1-6 digit house number + 3+ char street word |

**Blocking recall (on training set):**
- Candidate pairs generated: 38,421,902 pairs across training sources
- True pairs recalled: 7,511,834 / 7,638,365 (98.34%)
- Reduction ratio: 99.72% search space reduction

---

## 4. Matching / Similarity Model

**Features used (15 pairwise features):**

| Feature | Type | Notes |
|---------|------|-------|
| `name_token_sort_ratio` | RapidFuzz `token_sort_ratio` on `name_alphanum` | Handles word-order transpositions |
| `name_partial_ratio` | RapidFuzz `partial_ratio` on `name_alphanum` | Handles substring/DBA matches |
| `name_expanded_sort_ratio` | RapidFuzz `token_sort_ratio` on `name_expanded` | Abbreviation-aware |
| `name_qratio` | RapidFuzz `QRatio` on `name_alphanum` | Overall normalised edit distance |
| `name_jaro_winkler` | RapidFuzz `WRatio` on `name_alphanum` | Weighted ratio composite |
| `name_sorted_exact` | Binary exact match on `name_sorted` | High-precision exact gate |
| `name_alphanum_exact` | Binary exact match on `name_alphanum` | Raw exact gate |
| `addr_token_jaccard` | Jaccard similarity of address alphanum token sets | Address overlap |
| `addr_numeric_match` | Binary: first numeric address substring matches | Door/building/PIN number |
| `addr_jaro_winkler` | RapidFuzz `partial_ratio` on `addr_alphanum` | Address fuzzy similarity |
| `addr_prefix4_match` | Binary: first 4 chars of `addr_alphanum` match | Address prefix gate |
| `country_match` | Binary: `country` field string-equal | Open-set (France safe) |
| `name_len_ratio` | min/max of character lengths | Structural length similarity |
| `name_token_len_ratio` | min/max of token counts | Structural token count similarity |
| `name_prefix5_exact` | Binary: `name_prefix5` views exactly equal | Fast structural gate |

All 15 features are country-agnostic by design — no feature requires knowing or branching on the specific country label. The `country_match` feature uses raw string equality, so it naturally handles any country including unseen France.

**Model / Threshold:**
- **Model:** LightGBM binary classifier (MIT License, ~10M parameters << 8B ceiling)
  - `num_leaves=127`, `learning_rate=0.03`, `n_estimators=1000` (early stopped at iteration 597)
  - `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=0.1`
  - `scale_pos_weight` = neg/pos ratio (handles class imbalance from hard-negative mining)
  - Training data: Fold A (60% of S1 entities, ~1.32M entities)
  - Negatives: blocker hard-negatives (candidate pairs that are NOT true matches)
- **Threshold:** Grid-search over [0.10, 0.90] in steps of 0.01, optimising macro F_0.5 on fold B (10% S1 entities, entity-disjoint from fold A). **Optimal threshold: 0.4900**
- **Calibration:** Isotonic regression (`sklearn.isotonic.IsotonicRegression`) fitted on fold C (15% S1 entities), fold-disjoint from both training (fold A) and threshold tuning (fold B). Calibrated probabilities used for target-side deduplication.

---

## 5. Post-processing

1. **Threshold application**: candidate pairs with calibrated probability above the F_0.5-optimised threshold (0.4900) are predicted as matches.
2. **Singleton protection**: if no candidate pair passes threshold for an S1 entity, output an empty `matched_entity_ids` row. This correctly earns 1.0 macro F_0.5 for that entity.
3. **Target-side deduplication**: if multiple S1 entities claim the same S2/S3 target ID, assign the target to the S1 entity with the highest calibrated probability (greedy by score descending). This is valid because EDA confirmed target-side uniqueness (0 S2/S3 duplicates in ground truth).
4. **Candidate prioritisation**: `candidate_pairs.tsv` places matched IDs at index 0 for each S1 entity, followed by remaining blocking candidates (up to 100 per entity). This ensures every ID in `matching_results.tsv` appears in `candidate_pairs.tsv` (validator subset compliance).

---

## 6. Results

| Split | Precision | Recall | F_0.5 |
|-------|-----------|--------|-------|
| Fold B (Threshold Tuning, 10%) | 99.12% | 96.24% | 0.9843 |
| Fold D (Held-out Validation, 15%) | 98.85% | 83.20% | 0.9521 |
| Test Set (Expected Leaderboard) | ~97.5% | ~85.0% | ~0.9350 |

**Additional validation metrics:**
- LightGBM best iteration: 597 / 1000 (early stopped)
- Validation logloss: 0.0837
- Optimal F_0.5 threshold: 0.4900
- Test set singletons: 110,258 entities (6.36%), closely matching the 5.58% singleton rate in training ground truth
- Test set matched: 1,622,286 entities (93.64%)
- Total confident target links after target-side dedup: 6,223,555
- Submission validator: `utils/validate_submission.py` exit code 0 (PASS), 0 errors, 0 duplicate IDs, 100% candidate-match subset compliance
- `matching_results.tsv`: 1,732,544 rows (one per test S1 entity)
- `candidate_pairs.tsv`: 1,732,544 rows (one per test S1 entity)

---

## 7. Challenges & Lessons Learned

- **Scale**: 1.73M test S1 entities x 10.3M S2+S3 entities means O(10^13) pairwise comparisons. We use DuckDB in-memory SQL joins for blocking and chunked streaming for feature extraction (38,000+ pairs/sec) to stay within 16 GB RAM.
- **Recall ceiling vs. precision**: Initial 5-channel blocking achieved only 57.75% recall on fold D. Adding 4 additional channels (CH6-CH9: leetspeak, prefix, street, distinctive word) was critical to reaching 98.34% blocking recall while keeping the candidate set manageable.
- **France generalisation**: All features and blocking are country-agnostic (open string label). No code path branches on `{US, India}` only. The `country_match` feature naturally handles France through raw string equality.
- **Irreducible normaliser failures**: Non-Latin scripts (Telugu, Devanagari) and OCR-level typos (Sthoen, C0nstruction) cannot be resolved by normalisation alone. These cases rely on the edit-distance and fuzzy matching features (RapidFuzz token_sort_ratio, partial_ratio, WRatio) in the matching model.
- **F_0.5 precision bias**: Threshold was tuned specifically to maximise F_0.5 (not accuracy or F1). We deliberately prefer higher precision and accept missing some recall. Singletons score 1.0 when correctly identified as empty — the decision layer explicitly protects against false positive merges on singletons.
- **Target-side uniqueness**: EDA confirmed that each S2/S3 ID belongs to at most one S1 entity. This enables greedy target-side deduplication by calibrated confidence, which eliminates conflicting merges without transitive closure.
- **No external data**: All entity resolution is performed using only the provided training data. No commercial APIs, geocoding services, government databases, or internet sources were used.

---

## 8. References

- LightGBM (MIT License): https://github.com/microsoft/LightGBM
- RapidFuzz (MIT License): https://github.com/rapidfuzz/RapidFuzz
- scikit-learn IsotonicRegression (BSD 3-Clause): https://scikit-learn.org/stable/modules/isotonic.html
- sparse_dot_topn (Apache 2.0): https://github.com/ing-bank/sparse_dot_topn
- DuckDB (MIT License): https://duckdb.org/
- pandas (BSD 3-Clause): https://pandas.pydata.org/
- NumPy (BSD 3-Clause): https://numpy.org/
- SciPy (BSD 3-Clause): https://scipy.org/
- PyArrow (Apache 2.0): https://arrow.apache.org/docs/python/
