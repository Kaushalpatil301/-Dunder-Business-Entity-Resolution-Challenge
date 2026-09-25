from __future__ import annotations

import pickle
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 100_000

QUERY_SAMPLE_SIZE = 10_000

RANDOM_SEED = 42

MAX_FEATURES = 100_000

NGRAM_RANGE = (3, 5)

TOP_K_VALUES = [
    25,
    50,
    100,
    200,
]

MAX_TOP_K = max(TOP_K_VALUES)

ADDRESS_DF = 1_000

MIN_TOKEN_LENGTH = 3

NUMBER_PATTERN = re.compile(r"\d+")


# ============================================================
# PATHS
# ============================================================

def project_root() -> Path:

    return Path(__file__).resolve().parents[4]


def normalized_root() -> Path:

    return (
        project_root()
        / "data"
        / "normalized"
        / "train"
    )


def index_root() -> Path:

    return (
        project_root()
        / "data"
        / "blocking"
        / "indexes"
    )


def splits_root() -> Path:

    return (
        project_root()
        / "reports"
        / "splits"
    )


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_pickle(path: Path):

    with open(path, "rb") as file:
        return pickle.load(file)


def tokenize(value: str) -> set[str]:

    if not value:
        return set()

    tokens = re.findall(
        r"[a-z0-9]+",
        str(value).casefold(),
    )

    return {
        token
        for token in tokens
        if len(token) >= MIN_TOKEN_LENGTH
    }


def extract_numbers(value: str) -> set[str]:

    if not value:
        return set()

    return set(
        NUMBER_PATTERN.findall(
            str(value)
        )
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def load_truth(
    path: Path,
) -> dict[str, set[str]]:

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = {}

    for row in df.itertuples(
        index=False
    ):

        matched = row.matched_entity_ids

        if matched:

            truth[
                row.source1_entity_id
            ] = {
                x.strip()
                for x in matched.split(",")
                if x.strip()
            }

        else:

            truth[
                row.source1_entity_id
            ] = set()

    return truth


# ============================================================
# PHASE 4F INDEXES
# ============================================================

def load_phase4f_indexes():

    root = index_root()

    # --------------------------------------------------------
    # Name token
    # --------------------------------------------------------

    s2_name_payload = load_pickle(
        root / "s2_name_token.pkl"
    )

    s3_name_payload = load_pickle(
        root / "s3_name_token.pkl"
    )

    name_tokens = {
        "s2": s2_name_payload["index"],
        "s3": s3_name_payload["index"],
    }

    # --------------------------------------------------------
    # Exact address
    # --------------------------------------------------------

    exact_address = {
        "s2": load_pickle(
            root / "s2_address_exact.pkl"
        ),
        "s3": load_pickle(
            root / "s3_address_exact.pkl"
        ),
    }

    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    address_tokens = {
        "s2": load_pickle(
            root
            / f"s2_address_token_{ADDRESS_DF}.pkl"
        ),
        "s3": load_pickle(
            root
            / f"s3_address_token_{ADDRESS_DF}.pkl"
        ),
    }

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    numeric = {
        "s2": load_pickle(
            root / "s2_numeric.pkl"
        ),
        "s3": load_pickle(
            root / "s3_numeric.pkl"
        ),
    }

    return (
        name_tokens,
        exact_address,
        address_tokens,
        numeric,
    )


# ============================================================
# PHASE 4F CANDIDATES
# ============================================================

def phase4f_candidates(
    row,
    name_tokens,
    exact_address,
    address_tokens,
    numeric,
):

    candidates = set()

    # --------------------------------------------------------
    # Name token
    # --------------------------------------------------------

    for token in tokenize(
        row.name_norm
    ):

        candidates.update(
            name_tokens["s2"].get(
                token,
                [],
            )
        )

        candidates.update(
            name_tokens["s3"].get(
                token,
                [],
            )
        )

    address = row.address_norm

    # --------------------------------------------------------
    # Exact address
    # --------------------------------------------------------

    if address:

        candidates.update(
            exact_address["s2"].get(
                address,
                [],
            )
        )

        candidates.update(
            exact_address["s3"].get(
                address,
                [],
            )
        )

    # --------------------------------------------------------
    # Address tokens
    # --------------------------------------------------------

    for token in tokenize(
        address
    ):

        candidates.update(
            address_tokens["s2"].get(
                token,
                [],
            )
        )

        candidates.update(
            address_tokens["s3"].get(
                token,
                [],
            )
        )

    # --------------------------------------------------------
    # Numeric address
    # --------------------------------------------------------

    for number in extract_numbers(
        address
    ):

        candidates.update(
            numeric["s2"].get(
                number,
                [],
            )
        )

        candidates.update(
            numeric["s3"].get(
                number,
                [],
            )
        )

    return candidates


# ============================================================
# VALIDATION SAMPLE
# ============================================================

def load_validation_sample():

    root = normalized_root()

    s1_path = (
        root
        / "train_source1_normalized.tsv"
    )

    truth_path = (
        splits_root()
        / "validation_ground_truth.tsv"
    )

    truth = load_truth(
        truth_path
    )

    # --------------------------------------------------------
    # Collect only S1 entities with matches
    # --------------------------------------------------------

    chunks = []

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "name_norm",
            "address_norm",
        ],
        chunksize=CHUNK_SIZE,
    ):

        matched = chunk[
            chunk["entity_id"].isin(
                truth.keys()
            )
        ]

        if len(matched):

            chunks.append(
                matched
            )

    validation = pd.concat(
        chunks,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Deterministic sample
    # --------------------------------------------------------

    if len(validation) > QUERY_SAMPLE_SIZE:

        validation = validation.sample(
            n=QUERY_SAMPLE_SIZE,
            random_state=RANDOM_SEED,
        )

    validation = validation.reset_index(
        drop=True
    )

    print(
        f"Validation entities selected: "
        f"{len(validation):,}"
    )

    return validation, truth


# ============================================================
# FIT TF-IDF
# ============================================================

def fit_vectorizer():

    root = normalized_root()

    print()
    print("=" * 75)
    print("FITTING CHARACTER TF-IDF")
    print("=" * 75)

    samples = []

    # 250k from each source = 500k total.
    SAMPLE_PER_SOURCE = 250_000

    for source_name in [
        "train_source2_normalized.tsv",
        "train_source3_normalized.tsv",
    ]:

        path = root / source_name

        remaining = SAMPLE_PER_SOURCE

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=[
                "name_norm"
            ],
            chunksize=CHUNK_SIZE,
        ):

            take = min(
                remaining,
                len(chunk),
            )

            if take > 0:

                samples.extend(
                    chunk[
                        "name_norm"
                    ]
                    .iloc[:take]
                    .tolist()
                )

                remaining -= take

            if remaining <= 0:
                break

    print(
        f"Rows used for vocabulary: "
        f"{len(samples):,}"
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        lowercase=False,
        dtype=np.float32,
        sublinear_tf=True,
    )

    start = time.time()

    vectorizer.fit(
        samples
    )

    elapsed = (
        time.time()
        - start
    )

    print(
        f"Vocabulary size: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"Fit time: "
        f"{elapsed:.2f} seconds"
    )

    del samples

    return vectorizer


# ============================================================
# CONVERT SPARSE TOP-N RESULT TO DENSE ARRAYS
# ============================================================

def sparse_topn_to_dense(
    matrix,
    top_k,
):

    n_queries = matrix.shape[0]

    scores = np.zeros(
        (
            n_queries,
            top_k,
        ),
        dtype=np.float32,
    )

    indices = np.full(
        (
            n_queries,
            top_k,
        ),
        -1,
        dtype=np.int64,
    )

    for row_index in range(
        n_queries
    ):

        start = matrix.indptr[
            row_index
        ]

        end = matrix.indptr[
            row_index + 1
        ]

        row_indices = (
            matrix.indices[
                start:end
            ]
        )

        row_scores = (
            matrix.data[
                start:end
            ]
        )

        count = min(
            top_k,
            len(row_scores),
        )

        if count == 0:
            continue

        # sparse_dot_topn returns sorted
        # results when sort=True.
        scores[
            row_index,
            :count
        ] = row_scores[
            :count
        ]

        indices[
            row_index,
            :count
        ] = row_indices[
            :count
        ]

    return scores, indices


# ============================================================
# MERGE GLOBAL TOP-K
# ============================================================

def merge_topk(
    global_scores,
    global_ids,
    chunk_scores,
    chunk_ids,
    corpus_ids,
    top_k,
):

    n_queries = (
        global_scores.shape[0]
    )

    # --------------------------------------------------------
    # Convert chunk-local indexes to actual IDs.
    #
    # Fixed-width Unicode keeps memory bounded and avoids
    # Python dictionaries containing millions of strings.
    # --------------------------------------------------------

    chunk_entity_ids = np.empty(
        chunk_ids.shape,
        dtype="<U32",
    )

    valid = (
        chunk_ids >= 0
    )

    if np.any(valid):

        flat_indices = (
            chunk_ids[valid]
        )

        chunk_entity_ids[
            valid
        ] = np.asarray(
            corpus_ids,
            dtype="<U32",
        )[flat_indices]

    # --------------------------------------------------------
    # Merge each query's current top-K with this chunk's
    # top-K.
    # --------------------------------------------------------

    merged_scores = np.concatenate(
        [
            global_scores,
            chunk_scores,
        ],
        axis=1,
    )

    merged_ids = np.concatenate(
        [
            global_ids,
            chunk_entity_ids,
        ],
        axis=1,
    )

    # --------------------------------------------------------
    # Select top-K by score.
    # --------------------------------------------------------

    partition_indices = np.argpartition(
        merged_scores,
        -top_k,
        axis=1,
    )[
        :,
        -top_k:,
    ]

    row_indices = (
        np.arange(n_queries)[:, None]
    )

    selected_scores = (
        merged_scores[
            row_indices,
            partition_indices,
        ]
    )

    selected_ids = (
        merged_ids[
            row_indices,
            partition_indices,
        ]
    )

    # --------------------------------------------------------
    # Sort selected top-K descending.
    # --------------------------------------------------------

    order = np.argsort(
        selected_scores,
        axis=1,
    )[
        :,
        ::-1,
    ]

    final_scores = (
        np.take_along_axis(
            selected_scores,
            order,
            axis=1,
        )
    )

    final_ids = (
        np.take_along_axis(
            selected_ids,
            order,
            axis=1,
        )
    )

    return (
        final_scores.astype(
            np.float32,
            copy=False,
        ),
        final_ids,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "PHASE 4G - MEMORY-SAFE CHARACTER TF-IDF"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Load validation
    # --------------------------------------------------------

    validation, truth = (
        load_validation_sample()
    )

    n_queries = len(
        validation
    )

    # --------------------------------------------------------
    # Load Phase 4F indexes
    # --------------------------------------------------------

    print()
    print(
        "Loading Phase 4F indexes..."
    )

    (
        name_tokens,
        exact_address,
        address_tokens,
        numeric,
    ) = load_phase4f_indexes()

    print(
        "Phase 4F indexes loaded."
    )

    # --------------------------------------------------------
    # Compute Phase 4F baseline
    # --------------------------------------------------------

    print()
    print(
        "Computing Phase 4F baseline..."
    )

    phase4f_sets = []

    total_truth_links = 0
    phase4f_recovered = 0
    phase4f_complete = 0

    for row in validation.itertuples(
        index=False
    ):

        true_matches = truth[
            row.entity_id
        ]

        total_truth_links += len(
            true_matches
        )

        candidates = (
            phase4f_candidates(
                row,
                name_tokens,
                exact_address,
                address_tokens,
                numeric,
            )
        )

        phase4f_sets.append(
            candidates
        )

        recovered = (
            true_matches
            & candidates
        )

        phase4f_recovered += len(
            recovered
        )

        if recovered == true_matches:

            phase4f_complete += 1

    phase4f_recall = (
        phase4f_recovered
        / total_truth_links
    )

    print()
    print(
        f"Phase 4F sample recall: "
        f"{phase4f_recall:.6%}"
    )

    print(
        f"Phase 4F complete entities: "
        f"{phase4f_complete:,}"
    )

    # --------------------------------------------------------
    # Fit TF-IDF
    # --------------------------------------------------------

    vectorizer = fit_vectorizer()

    query_texts = (
        validation[
            "name_norm"
        ]
        .tolist()
    )

    print()
    print(
        "Transforming validation queries..."
    )

    query_matrix = (
        vectorizer.transform(
            query_texts
        )
        .tocsr()
    )

    print(
        f"Query matrix: "
        f"{query_matrix.shape}"
    )

    # --------------------------------------------------------
    # Global top-K storage
    # --------------------------------------------------------

    # Fixed-size arrays:
    #
    # 10,000 x 200
    #
    # This is dramatically smaller than storing Python
    # dictionaries of every candidate.

    global_scores = np.zeros(
        (
            n_queries,
            MAX_TOP_K,
        ),
        dtype=np.float32,
    )

    global_ids = np.full(
        (
            n_queries,
            MAX_TOP_K,
        ),
        "",
        dtype="<U32",
    )

    # --------------------------------------------------------
    # Process S2 + S3 in chunks
    # --------------------------------------------------------

    root = normalized_root()

    corpus_start = time.time()

    total_corpus_rows = 0

    for source_name in [
        "train_source2_normalized.tsv",
        "train_source3_normalized.tsv",
    ]:

        path = root / source_name

        print()
        print("=" * 75)
        print(
            f"PROCESSING {source_name}"
        )
        print("=" * 75)

        source_rows = 0

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "name_norm",
            ],
            chunksize=CHUNK_SIZE,
        ):

            source_rows += len(
                chunk
            )

            total_corpus_rows += len(
                chunk
            )

            print(
                f"Processed source rows: "
                f"{source_rows:,} "
                f"| total: "
                f"{total_corpus_rows:,}"
            )

            corpus_names = (
                chunk[
                    "name_norm"
                ]
                .tolist()
            )

            corpus_ids = (
                chunk[
                    "entity_id"
                ]
                .tolist()
            )

            # --------------------------------------------
            # TF-IDF transform
            # --------------------------------------------

            corpus_matrix = (
                vectorizer.transform(
                    corpus_names
                )
                .tocsr()
            )

            # --------------------------------------------
            # Sparse top-N multiplication
            # --------------------------------------------

            similarities = (
                sp_matmul_topn(
                    query_matrix,
                    corpus_matrix.T,
                    top_n=MAX_TOP_K,
                    threshold=0.0,
                    sort=True,
                )
            )

            # --------------------------------------------
            # Convert only top-K to dense arrays
            # --------------------------------------------

            chunk_scores, chunk_indices = (
                sparse_topn_to_dense(
                    similarities,
                    MAX_TOP_K,
                )
            )

            # --------------------------------------------
            # Merge with global top-K
            # --------------------------------------------

            (
                global_scores,
                global_ids,
            ) = merge_topk(
                global_scores,
                global_ids,
                chunk_scores,
                chunk_indices,
                corpus_ids,
                MAX_TOP_K,
            )

            # --------------------------------------------
            # Release chunk memory
            # --------------------------------------------

            del corpus_matrix
            del similarities
            del chunk_scores
            del chunk_indices

    elapsed = (
        time.time()
        - corpus_start
    )

    print()
    print(
        f"Total corpus rows processed: "
        f"{total_corpus_rows:,}"
    )

    print(
        f"Total TF-IDF retrieval time: "
        f"{elapsed / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # Evaluate different K values
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("PHASE 4G RESULTS")
    print("=" * 75)

    print()

    print(
        f"{'K':>8} "
        f"{'TFIDF Recall':>15} "
        f"{'Union Recall':>15} "
        f"{'Added TP':>12} "
        f"{'Complete':>12} "
        f"{'Mean Union':>14} "
        f"{'Median':>12} "
        f"{'P95':>12}"
    )

    print("-" * 120)

    for top_k in TOP_K_VALUES:

        tfidf_recovered = 0
        union_recovered = 0
        added_true_positives = 0
        complete_union = 0

        candidate_counts = []

        for query_index, row in enumerate(
            validation.itertuples(
                index=False
            )
        ):

            true_matches = truth[
                row.entity_id
            ]

            tfidf_candidates = {
                candidate_id
                for candidate_id in (
                    global_ids[
                        query_index,
                        :top_k,
                    ]
                )
                if candidate_id
            }

            phase4f = (
                phase4f_sets[
                    query_index
                ]
            )

            union_candidates = (
                phase4f
                | tfidf_candidates
            )

            tfidf_hits = (
                true_matches
                & tfidf_candidates
            )

            union_hits = (
                true_matches
                & union_candidates
            )

            new_true_positives = (
                true_matches
                & tfidf_candidates
                - phase4f
            )

            tfidf_recovered += len(
                tfidf_hits
            )

            union_recovered += len(
                union_hits
            )

            added_true_positives += len(
                new_true_positives
            )

            if (
                true_matches
                <= union_candidates
            ):

                complete_union += 1

            candidate_counts.append(
                len(union_candidates)
            )

        tfidf_recall = (
            tfidf_recovered
            / total_truth_links
        )

        union_recall = (
            union_recovered
            / total_truth_links
        )

        series = pd.Series(
            candidate_counts
        )

        print(
            f"{top_k:>8} "
            f"{tfidf_recall:>14.6%} "
            f"{union_recall:>14.6%} "
            f"{added_true_positives:>12,} "
            f"{complete_union:>12,} "
            f"{series.mean():>14.2f} "
            f"{series.median():>12.2f} "
            f"{series.quantile(.95):>12.2f}"
        )

    print()
    print("=" * 75)
    print(
        "PHASE 4G COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()