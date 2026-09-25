from __future__ import annotations

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

QUERY_SAMPLE_SIZE = 50_000

RANDOM_SEED = 42

MAX_FEATURES = 200_000

NGRAM_RANGE = (3, 5)

TOP_K_VALUES = [
    25,
    50,
    100,
    200,
    500,
]

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
# BASIC HELPERS
# ============================================================

def load_pickle(path: Path):

    import pickle

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
# TRUTH
# ============================================================

def load_truth(path: Path) -> dict[str, set[str]]:

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = {}

    for row in df.itertuples(index=False):

        matches = row.matched_entity_ids

        if matches:

            truth[row.source1_entity_id] = {
                x.strip()
                for x in matches.split(",")
                if x.strip()
            }

        else:

            truth[row.source1_entity_id] = set()

    return truth


# ============================================================
# PHASE 4F BLOCKERS
# ============================================================

def load_phase4f_indexes():

    root = index_root()

    # --------------------------------------------
    # Name token
    # --------------------------------------------

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

    # --------------------------------------------
    # Exact address
    # --------------------------------------------

    exact_address = {
        "s2": load_pickle(
            root / "s2_address_exact.pkl"
        ),
        "s3": load_pickle(
            root / "s3_address_exact.pkl"
        ),
    }

    # --------------------------------------------
    # Address token
    # --------------------------------------------

    address_tokens = {
        "s2": load_pickle(
            root / f"s2_address_token_{ADDRESS_DF}.pkl"
        ),
        "s3": load_pickle(
            root / f"s3_address_token_{ADDRESS_DF}.pkl"
        ),
    }

    # --------------------------------------------
    # Numeric
    # --------------------------------------------

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


def phase4f_candidates(
    row,
    name_tokens,
    exact_address,
    address_tokens,
    numeric,
):

    candidates = set()

    # --------------------------------------------
    # Name tokens
    # --------------------------------------------

    for token in tokenize(row.name_norm):

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

    # --------------------------------------------
    # Exact address
    # --------------------------------------------

    address = row.address_norm

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

    # --------------------------------------------
    # Address tokens
    # --------------------------------------------

    for token in tokenize(address):

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

    # --------------------------------------------
    # Numbers
    # --------------------------------------------

    for number in extract_numbers(address):

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
# LOAD VALIDATION QUERIES
# ============================================================

def load_validation_queries():

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

    rows = []

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

            rows.append(
                matched
            )

    validation = pd.concat(
        rows,
        ignore_index=True,
    )

    validation = validation[
        validation["entity_id"].map(
            lambda x: bool(truth.get(x))
        )
    ]

    # Deterministic sample.
    if len(validation) > QUERY_SAMPLE_SIZE:

        validation = validation.sample(
            n=QUERY_SAMPLE_SIZE,
            random_state=RANDOM_SEED,
        )

    validation = validation.reset_index(
        drop=True
    )

    print(
        f"Validation matched entities selected: "
        f"{len(validation):,}"
    )

    return validation, truth


# ============================================================
# CORPUS ITERATOR
# ============================================================

def corpus_chunks():

    root = normalized_root()

    for source_name in [
        "train_source2_normalized.tsv",
        "train_source3_normalized.tsv",
    ]:

        path = root / source_name

        print()
        print(
            f"Reading corpus source: "
            f"{source_name}"
        )

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

            yield chunk


# ============================================================
# FIT TF-IDF VOCABULARY
# ============================================================

def fit_vectorizer():

    root = normalized_root()

    print()
    print("=" * 75)
    print("FITTING CHARACTER TF-IDF")
    print("=" * 75)

    # We use a deterministic 1M-row fitting sample
    # to keep the vocabulary construction practical.
    samples = []

    sample_per_source = 500_000

    for source_name in [
        "train_source2_normalized.tsv",
        "train_source3_normalized.tsv",
    ]:

        path = root / source_name

        remaining = sample_per_source

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=["name_norm"],
            chunksize=CHUNK_SIZE,
        ):

            take = min(
                remaining,
                len(chunk),
            )

            if take > 0:

                samples.extend(
                    chunk["name_norm"]
                    .iloc[:take]
                    .tolist()
                )

                remaining -= take

            if remaining <= 0:
                break

    print(
        f"Rows used to fit vocabulary: "
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

    vectorizer.fit(samples)

    elapsed = time.time() - start

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
# QUERY MATRIX
# ============================================================

def transform_queries(
    vectorizer,
    validation,
):

    names = validation[
        "name_norm"
    ].tolist()

    return vectorizer.transform(
        names
    ).tocsr()


# ============================================================
# TOP-K RETRIEVAL
# ============================================================

def retrieve_topk_for_corpus_chunk(
    query_matrix,
    corpus_matrix,
    corpus_ids,
    top_k,
):

    similarities = sp_matmul_topn(
        query_matrix,
        corpus_matrix.T,
        top_n=top_k,
        threshold=0.0,
        sort=True,
    )

    result = []

    for row_index in range(
        similarities.shape[0]
    ):

        start = similarities.indptr[
            row_index
        ]

        end = similarities.indptr[
            row_index + 1
        ]

        indices = similarities.indices[
            start:end
        ]

        scores = similarities.data[
            start:end
        ]

        pairs = [
            (
                corpus_ids[index],
                float(score),
            )
            for index, score in zip(
                indices,
                scores,
            )
        ]

        result.append(pairs)

    return result


# ============================================================
# MAIN TF-IDF EVALUATION
# ============================================================

def main():

    print("=" * 75)
    print(
        "PHASE 4G - CHARACTER TF-IDF UNION"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Load validation sample
    # --------------------------------------------------------

    validation, truth = (
        load_validation_queries()
    )

    # --------------------------------------------------------
    # Load Phase 4F indexes
    # --------------------------------------------------------

    print()
    print("Loading Phase 4F indexes...")

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
    # Compute Phase 4F candidates once
    # --------------------------------------------------------

    print()
    print(
        "Generating Phase 4F baseline candidates..."
    )

    phase4f_sets = []

    phase4f_recovered = 0
    phase4f_complete = 0
    phase4f_total_truth = 0

    for row in validation.itertuples(
        index=False
    ):

        true_matches = truth[
            row.entity_id
        ]

        phase4f_total_truth += len(
            true_matches
        )

        candidates = phase4f_candidates(
            row,
            name_tokens,
            exact_address,
            address_tokens,
            numeric,
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

    baseline_recall = (
        phase4f_recovered
        / phase4f_total_truth
    )

    print()
    print(
        f"Phase 4F baseline recall: "
        f"{baseline_recall:.6%}"
    )

    print(
        f"Phase 4F complete entities: "
        f"{phase4f_complete:,}"
    )

    # --------------------------------------------------------
    # Fit TF-IDF
    # --------------------------------------------------------

    vectorizer = fit_vectorizer()

    query_matrix = transform_queries(
        vectorizer,
        validation,
    )

    print()
    print(
        f"Query matrix shape: "
        f"{query_matrix.shape}"
    )

    # --------------------------------------------------------
    # Evaluation storage
    # --------------------------------------------------------

    results = {}

    for top_k in TOP_K_VALUES:

        results[top_k] = {
            "recovered": 0,
            "complete": 0,
            "candidate_counts": [],
        }

    # --------------------------------------------------------
    # Process corpus in chunks
    # --------------------------------------------------------

    # For every query we maintain only its current
    # best TOP_K results across corpus chunks.

    best_candidates = {
        top_k: [
            {}
            for _ in range(len(validation))
        ]
        for top_k in TOP_K_VALUES
    }

    total_corpus_rows = 0

    corpus_start = time.time()

    for corpus_chunk in corpus_chunks():

        total_corpus_rows += len(
            corpus_chunk
        )

        print()
        print(
            f"TF-IDF corpus rows processed: "
            f"{total_corpus_rows:,}"
        )

        corpus_names = (
            corpus_chunk[
                "name_norm"
            ].tolist()
        )

        corpus_ids = (
            corpus_chunk[
                "entity_id"
            ].tolist()
        )

        corpus_matrix = (
            vectorizer.transform(
                corpus_names
            ).tocsr()
        )

        # ----------------------------------------------------
        # Retrieve top 500.
        #
        # Smaller K values are obtained by taking prefixes
        # of the same ranking.
        # ----------------------------------------------------

        retrieved = (
            retrieve_topk_for_corpus_chunk(
                query_matrix=query_matrix,
                corpus_matrix=corpus_matrix,
                corpus_ids=corpus_ids,
                top_k=max(TOP_K_VALUES),
            )
        )

        for query_index, pairs in enumerate(
            retrieved
        ):

            for candidate_id, score in pairs:

                for top_k in TOP_K_VALUES:

                    current = best_candidates[
                        top_k
                    ][query_index]

                    previous = current.get(
                        candidate_id
                    )

                    if (
                        previous is None
                        or score > previous
                    ):

                        current[
                            candidate_id
                        ] = score

        del corpus_matrix

    corpus_elapsed = (
        time.time()
        - corpus_start
    )

    print()
    print(
        f"Total corpus rows processed: "
        f"{total_corpus_rows:,}"
    )

    print(
        f"Total retrieval time: "
        f"{corpus_elapsed / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # Evaluate union
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print("PHASE 4G RESULTS")
    print("=" * 75)

    print()
    print(
        f"{'Top-K':>10} "
        f"{'TFIDF Recall':>15} "
        f"{'Union Recall':>15} "
        f"{'Complete':>14} "
        f"{'Mean Cand.':>14} "
        f"{'Median':>12} "
        f"{'P95':>12}"
    )

    print("-" * 105)

    for top_k in TOP_K_VALUES:

        recovered_tfidf = 0
        recovered_union = 0
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

            ranked = best_candidates[
                top_k
            ][query_index]

            tfidf_candidates = set(
                ranked.keys()
            )

            union_candidates = (
                phase4f_sets[
                    query_index
                ]
                | tfidf_candidates
            )

            recovered_tfidf += len(
                true_matches
                & tfidf_candidates
            )

            recovered_union += len(
                true_matches
                & union_candidates
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
            recovered_tfidf
            / phase4f_total_truth
        )

        union_recall = (
            recovered_union
            / phase4f_total_truth
        )

        series = pd.Series(
            candidate_counts
        )

        print(
            f"{top_k:>10} "
            f"{tfidf_recall:>14.6%} "
            f"{union_recall:>14.6%} "
            f"{complete_union:>14,} "
            f"{series.mean():>14.2f} "
            f"{series.median():>12.2f} "
            f"{series.quantile(.95):>12.2f}"
        )

        results[
            top_k
        ]["recovered"] = recovered_union

        results[
            top_k
        ]["complete"] = complete_union

    print()
    print("=" * 75)
    print("PHASE 4G COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()