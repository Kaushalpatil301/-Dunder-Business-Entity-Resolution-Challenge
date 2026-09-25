from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


CHUNK_SIZE = 50_000

CORPUS_ROWS_PER_SOURCE = 100_000
VALIDATION_QUERY_ROWS = 10_000

NGRAM_RANGE = (3, 5)
MAX_FEATURES = 200_000

QUERY_BATCH_SIZE = 100

RANK_THRESHOLDS = [
    10,
    25,
    50,
    100,
    200,
    500,
]


def load_names(
    path: Path,
    limit: int,
) -> tuple[list[str], list[str]]:

    entity_ids = []
    names = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "name_norm"],
        chunksize=CHUNK_SIZE,
    ):

        entity_ids.extend(
            chunk["entity_id"].tolist()
        )

        names.extend(
            chunk["name_norm"].tolist()
        )

        if len(names) >= limit:
            break

    return (
        entity_ids[:limit],
        names[:limit],
    )


def load_validation(
    path: Path,
    limit: int,
) -> tuple[list[str], list[str]]:

    entity_ids = []
    names = []

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

        for row in chunk.itertuples(
            index=False
        ):

            entity_ids.append(
                row.entity_id
            )

            names.append(
                row.name_norm
            )

            if len(entity_ids) >= limit:
                break

        if len(entity_ids) >= limit:
            break

    return (
        entity_ids,
        names,
    )


def load_ground_truth(
    path: Path,
    query_ids: set[str],
) -> dict[str, set[str]]:

    truth = {}

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    for row in df.itertuples(
        index=False
    ):

        s1_id = row.source1_entity_id

        if s1_id not in query_ids:
            continue

        matches = row.matched_entity_ids

        if not matches:
            truth[s1_id] = set()
            continue

        truth[s1_id] = {
            match_id.strip()
            for match_id in matches.split(",")
            if match_id.strip()
        }

    return truth


def calculate_true_match_ranks(
    similarity_row,
    corpus_ids: list[str],
    true_matches: set[str],
) -> list[int]:

    if not true_matches:
        return []

    scores = similarity_row.data
    indices = similarity_row.indices

    if len(scores) == 0:
        return []

    # Sort every non-zero similarity in descending order.
    order = np.argsort(
        scores
    )[::-1]

    ranks = []

    for position in range(
        len(order)
    ):

        corpus_index = indices[
            order[position]
        ]

        candidate_id = corpus_ids[
            corpus_index
        ]

        if candidate_id in true_matches:

            ranks.append(
                position + 1
            )

    return ranks


def main() -> None:

    project_root = (
        Path(__file__).resolve().parents[4]
    )

    normalized_root = (
        project_root
        / "data"
        / "normalized"
        / "train"
    )

    splits_root = (
        project_root
        / "reports"
        / "splits"
    )

    s2_path = (
        normalized_root
        / "train_source2_normalized.tsv"
    )

    s3_path = (
        normalized_root
        / "train_source3_normalized.tsv"
    )

    s1_path = (
        normalized_root
        / "train_source1_normalized.tsv"
    )

    truth_path = (
        splits_root
        / "validation_ground_truth.tsv"
    )

    print("=" * 70)
    print("PHASE 4D.2 - TRUE-MATCH TF-IDF RANK TEST")
    print("=" * 70)

    print()
    print(
        f"S2 corpus rows: "
        f"{CORPUS_ROWS_PER_SOURCE:,}"
    )

    print(
        f"S3 corpus rows: "
        f"{CORPUS_ROWS_PER_SOURCE:,}"
    )

    print(
        f"Validation S1 queries: "
        f"{VALIDATION_QUERY_ROWS:,}"
    )

    # --------------------------------------------------
    # Load corpus
    # --------------------------------------------------

    start = time.time()

    print()
    print("Loading S2...")

    s2_ids, s2_names = load_names(
        s2_path,
        CORPUS_ROWS_PER_SOURCE,
    )

    print(
        f"S2 records: "
        f"{len(s2_ids):,}"
    )

    print("Loading S3...")

    s3_ids, s3_names = load_names(
        s3_path,
        CORPUS_ROWS_PER_SOURCE,
    )

    print(
        f"S3 records: "
        f"{len(s3_ids):,}"
    )

    corpus_ids = (
        s2_ids + s3_ids
    )

    corpus_names = (
        s2_names + s3_names
    )

    corpus_id_set = set(
        corpus_ids
    )

    print(
        f"Total corpus: "
        f"{len(corpus_ids):,}"
    )

    print(
        f"Loading time: "
        f"{time.time() - start:.2f}s"
    )

    # --------------------------------------------------
    # Load validation
    # --------------------------------------------------

    print()
    print("Loading validation S1...")

    validation_ids, validation_names = (
        load_validation(
            s1_path,
            VALIDATION_QUERY_ROWS,
        )
    )

    truth = load_ground_truth(
        truth_path,
        set(validation_ids),
    )

    print(
        f"Validation queries: "
        f"{len(validation_ids):,}"
    )

    print(
        f"Queries with truth: "
        f"{len(truth):,}"
    )

    # --------------------------------------------------
    # Determine true matches actually present
    # --------------------------------------------------

    eligible_queries = []

    eligible_truth = {}

    total_truth_links = 0
    present_truth_links = 0

    for index, s1_id in enumerate(
        validation_ids
    ):

        true_matches = truth.get(
            s1_id,
            set(),
        )

        present_matches = (
            true_matches
            & corpus_id_set
        )

        if not present_matches:
            continue

        eligible_queries.append(
            index
        )

        eligible_truth[s1_id] = (
            present_matches
        )

        total_truth_links += len(
            true_matches
        )

        present_truth_links += len(
            present_matches
        )

    print()
    print(
        f"Queries with ≥1 true match "
        f"in corpus: "
        f"{len(eligible_queries):,}"
    )

    print(
        f"True links present in corpus: "
        f"{present_truth_links:,}"
    )

    print(
        f"Total true links for queries: "
        f"{total_truth_links:,}"
    )

    if not eligible_queries:
        print(
            "No eligible queries. Exiting."
        )
        return

    # --------------------------------------------------
    # Fit TF-IDF
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("FITTING CHARACTER TF-IDF")
    print("=" * 70)

    start = time.time()

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=NGRAM_RANGE,
        min_df=2,
        max_features=MAX_FEATURES,
        lowercase=False,
        dtype=np.float32,
        sublinear_tf=True,
        norm="l2",
    )

    corpus_matrix = (
        vectorizer.fit_transform(
            corpus_names
        )
    )

    matrix_memory = (
        corpus_matrix.data.nbytes
        + corpus_matrix.indices.nbytes
        + corpus_matrix.indptr.nbytes
    )

    print(
        f"Matrix shape: "
        f"{corpus_matrix.shape}"
    )

    print(
        f"Non-zero values: "
        f"{corpus_matrix.nnz:,}"
    )

    print(
        f"Features: "
        f"{len(vectorizer.vocabulary_):,}"
    )

    print(
        f"Matrix memory: "
        f"{matrix_memory / (1024 ** 2):.2f} MB"
    )

    print(
        f"TF-IDF time: "
        f"{time.time() - start:.2f}s"
    )

    # --------------------------------------------------
    # Transform only eligible queries
    # --------------------------------------------------

    eligible_names = [
        validation_names[index]
        for index in eligible_queries
    ]

    eligible_ids = [
        validation_ids[index]
        for index in eligible_queries
    ]

    query_matrix = vectorizer.transform(
        eligible_names
    )

    # --------------------------------------------------
    # Rank true matches
    # --------------------------------------------------

    rank_values = []

    processed = 0

    print()
    print("=" * 70)
    print("CALCULATING TRUE-MATCH RANKS")
    print("=" * 70)

    start = time.time()

    for batch_start in range(
        0,
        query_matrix.shape[0],
        QUERY_BATCH_SIZE,
    ):

        batch_end = min(
            batch_start + QUERY_BATCH_SIZE,
            query_matrix.shape[0],
        )

        query_batch = query_matrix[
            batch_start:batch_end
        ]

        similarity = (
            query_batch
            @ corpus_matrix.T
        )

        for local_index in range(
            similarity.shape[0]
        ):

            global_index = (
                batch_start
                + local_index
            )

            s1_id = eligible_ids[
                global_index
            ]

            true_matches = (
                eligible_truth[s1_id]
            )

            ranks = (
                calculate_true_match_ranks(
                    similarity.getrow(
                        local_index
                    ),
                    corpus_ids,
                    true_matches,
                )
            )

            rank_values.extend(
                ranks
            )

        processed += (
            batch_end - batch_start
        )

        print(
            f"Processed eligible queries: "
            f"{processed:,}/"
            f"{len(eligible_queries):,}"
        )

        del similarity

    elapsed = time.time() - start

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    ranks = np.array(
        rank_values,
        dtype=np.int64,
    )

    print()
    print("=" * 70)
    print("PHASE 4D.2 RESULTS")
    print("=" * 70)

    print(
        f"True matches ranked: "
        f"{len(ranks):,}"
    )

    print(
        f"Rank calculation time: "
        f"{elapsed:.2f}s"
    )

    if len(ranks) == 0:
        print(
            "No true matches received "
            "a non-zero TF-IDF similarity."
        )
        return

    print()
    print(
        f"Mean rank: "
        f"{ranks.mean():.2f}"
    )

    print(
        f"Median rank: "
        f"{np.median(ranks):.2f}"
    )

    print(
        f"P95 rank: "
        f"{np.percentile(ranks, 95):.2f}"
    )

    print(
        f"Maximum rank: "
        f"{ranks.max():,}"
    )

    print()
    print(
        f"{'Rank threshold':>18} "
        f"{'True links':>15} "
        f"{'Recall':>15}"
    )

    print("-" * 52)

    for threshold in RANK_THRESHOLDS:

        recovered = np.sum(
            ranks <= threshold
        )

        recall = (
            recovered / len(ranks)
        )

        print(
            f"{threshold:>18,} "
            f"{recovered:>15,} "
            f"{recall:>14.6%}"
        )

    print()
    print("=" * 70)
    print("PHASE 4D.2 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()