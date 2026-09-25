from __future__ import annotations

import pickle
import re
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 50_000

QUERY_SAMPLE_SIZE = 5_000

CORPUS_PER_SOURCE = 500_000

RANDOM_SEED = 42

TOP_K_VALUES = [
    25,
    50,
    100,
    200,
]

EMBEDDING_BATCH_SIZE = 64

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


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
# PICKLE
# ============================================================

def load_pickle(path: Path):

    with open(path, "rb") as file:
        return pickle.load(file)


# ============================================================
# TEXT PREPARATION
# ============================================================

def build_embedding_text(
    name: str,
    address: str,
    country: str = "",
) -> str:

    name = str(name).strip()
    address = str(address).strip()
    country = str(country).strip()

    # Name receives the strongest emphasis.
    #
    # Repeating the name gives the embedding model a stronger
    # signal for entity identity than the noisy address.
    #
    # Address is retained because it can disambiguate common
    # business names.

    return (
        f"business name: {name} "
        f"business name: {name} "
        f"address: {address} "
        f"country: {country}"
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def load_truth():

    path = (
        splits_root()
        / "validation_ground_truth.tsv"
    )

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
# VALIDATION SAMPLE
# ============================================================

def load_validation_sample():

    root = normalized_root()

    path = (
        root
        / "train_source1_normalized.tsv"
    )

    truth = load_truth()

    chunks = []

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "name_norm",
            "address_norm",
            "country_norm",
        ],
        chunksize=CHUNK_SIZE,
    ):

        # Only entities with at least one match.
        chunk = chunk[
            chunk["entity_id"].isin(
                truth.keys()
            )
        ]

        if len(chunk):

            chunks.append(
                chunk
            )

    validation = pd.concat(
        chunks,
        ignore_index=True,
    )

    validation = validation[
        validation["entity_id"].map(
            lambda x: bool(
                truth.get(x)
            )
        )
    ]

    if len(validation) > QUERY_SAMPLE_SIZE:

        validation = validation.sample(
            n=QUERY_SAMPLE_SIZE,
            random_state=RANDOM_SEED,
        )

    validation = validation.reset_index(
        drop=True
    )

    print(
        f"Validation entities: "
        f"{len(validation):,}"
    )

    return validation, truth


# ============================================================
# PHASE 4F INDEXES
# ============================================================

def load_phase4f_indexes():

    root = index_root()

    name_s2 = load_pickle(
        root / "s2_name_token.pkl"
    )

    name_s3 = load_pickle(
        root / "s3_name_token.pkl"
    )

    exact_s2 = load_pickle(
        root / "s2_address_exact.pkl"
    )

    exact_s3 = load_pickle(
        root / "s3_address_exact.pkl"
    )

    address_s2 = load_pickle(
        root / "s2_address_token_1000.pkl"
    )

    address_s3 = load_pickle(
        root / "s3_address_token_1000.pkl"
    )

    numeric_s2 = load_pickle(
        root / "s2_numeric.pkl"
    )

    numeric_s3 = load_pickle(
        root / "s3_numeric.pkl"
    )

    return (
        name_s2["index"],
        name_s3["index"],
        exact_s2,
        exact_s3,
        address_s2,
        address_s3,
        numeric_s2,
        numeric_s3,
    )


# ============================================================
# PHASE 4F CANDIDATES
# ============================================================

def tokenize(value):

    if not value:

        return set()

    return {
        x
        for x in re.findall(
            r"[a-z0-9]+",
            str(value).casefold(),
        )
        if len(x) >= 3
    }


def numbers(value):

    if not value:

        return set()

    return set(
        re.findall(
            r"\d+",
            str(value),
        )
    )


def phase4f_candidates(
    row,
    name_s2,
    name_s3,
    exact_s2,
    exact_s3,
    address_s2,
    address_s3,
    numeric_s2,
    numeric_s3,
):

    candidates = set()

    # --------------------------------------------------------
    # Name tokens
    # --------------------------------------------------------

    for token in tokenize(
        row.name_norm
    ):

        candidates.update(
            name_s2.get(
                token,
                [],
            )
        )

        candidates.update(
            name_s3.get(
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
            exact_s2.get(
                address,
                [],
            )
        )

        candidates.update(
            exact_s3.get(
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
            address_s2.get(
                token,
                [],
            )
        )

        candidates.update(
            address_s3.get(
                token,
                [],
            )
        )

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    for number in numbers(
        address
    ):

        candidates.update(
            numeric_s2.get(
                number,
                [],
            )
        )

        candidates.update(
            numeric_s3.get(
                number,
                [],
            )
        )

    return candidates


# ============================================================
# LOAD CORPUS SAMPLE
# ============================================================

def load_corpus_sample():

    root = normalized_root()

    frames = []

    for source_number in [2, 3]:

        path = (
            root
            / f"train_source{source_number}_normalized.tsv"
        )

        remaining = CORPUS_PER_SOURCE

        print()
        print(
            f"Loading source {source_number} "
            f"corpus sample..."
        )

        for chunk in pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "name_norm",
                "address_norm",
                "country_norm",
            ],
            chunksize=CHUNK_SIZE,
        ):

            take = min(
                remaining,
                len(chunk),
            )

            if take:

                frames.append(
                    chunk.iloc[:take]
                )

                remaining -= take

            if remaining <= 0:

                break

        print(
            f"Loaded "
            f"{CORPUS_PER_SOURCE - remaining:,}"
            f" source {source_number} records."
        )

    corpus = pd.concat(
        frames,
        ignore_index=True,
    )

    print()
    print(
        f"Total ANN corpus: "
        f"{len(corpus):,}"
    )

    return corpus


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "PHASE 4H - DENSE MULTILINGUAL ANN PILOT"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation, truth = (
        load_validation_sample()
    )

    # --------------------------------------------------------
    # Phase 4F baseline
    # --------------------------------------------------------

    print()
    print(
        "Loading Phase 4F indexes..."
    )

    (
        name_s2,
        name_s3,
        exact_s2,
        exact_s3,
        address_s2,
        address_s3,
        numeric_s2,
        numeric_s3,
    ) = load_phase4f_indexes()

    print(
        "Phase 4F indexes loaded."
    )

    phase4f_sets = []

    total_truth = 0
    phase4f_recovered = 0
    phase4f_complete = 0

    for row in validation.itertuples(
        index=False
    ):

        truth_set = truth[
            row.entity_id
        ]

        total_truth += len(
            truth_set
        )

        candidates = (
            phase4f_candidates(
                row,
                name_s2,
                name_s3,
                exact_s2,
                exact_s3,
                address_s2,
                address_s3,
                numeric_s2,
                numeric_s3,
            )
        )

        phase4f_sets.append(
            candidates
        )

        recovered = (
            truth_set
            & candidates
        )

        phase4f_recovered += len(
            recovered
        )

        if recovered == truth_set:

            phase4f_complete += 1

    print()
    print(
        f"Phase 4F sample recall: "
        f"{phase4f_recovered / total_truth:.6%}"
    )

    print(
        f"Phase 4F complete: "
        f"{phase4f_complete:,}"
    )

    # --------------------------------------------------------
    # Load corpus
    # --------------------------------------------------------

    corpus = load_corpus_sample()

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "LOADING MULTILINGUAL EMBEDDING MODEL"
    )
    print("=" * 75)

    print(
        f"Model: {MODEL_NAME}"
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    # --------------------------------------------------------
    # Build corpus text
    # --------------------------------------------------------

    print()
    print(
        "Preparing corpus text..."
    )

    corpus_text = [
        build_embedding_text(
            name,
            address,
            country,
        )
        for name, address, country
        in zip(
            corpus["name_norm"],
            corpus["address_norm"],
            corpus["country_norm"],
        )
    ]

    # --------------------------------------------------------
    # Encode corpus
    # --------------------------------------------------------

    print()
    print(
        "Encoding corpus..."
    )

    start = time.time()

    corpus_embeddings = model.encode(
        corpus_text,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed = time.time() - start

    corpus_embeddings = (
        np.asarray(
            corpus_embeddings,
            dtype=np.float32,
        )
    )

    print(
        f"Corpus embedding shape: "
        f"{corpus_embeddings.shape}"
    )

    print(
        f"Corpus encoding time: "
        f"{elapsed / 60:.2f} minutes"
    )

    del corpus_text

    # --------------------------------------------------------
    # Build FAISS index
    # --------------------------------------------------------

    print()
    print(
        "Building FAISS index..."
    )

    dimension = (
        corpus_embeddings.shape[1]
    )

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        corpus_embeddings
    )

    print(
        f"FAISS vectors: "
        f"{index.ntotal:,}"
    )

    # --------------------------------------------------------
    # Encode queries
    # --------------------------------------------------------

    print()
    print(
        "Encoding validation queries..."
    )

    query_text = [
        build_embedding_text(
            name,
            address,
            country,
        )
        for name, address, country
        in zip(
            validation["name_norm"],
            validation["address_norm"],
            validation["country_norm"],
        )
    ]

    query_embeddings = model.encode(
        query_text,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    query_embeddings = (
        np.asarray(
            query_embeddings,
            dtype=np.float32,
        )
    )

    del query_text

    # --------------------------------------------------------
    # ANN search
    # --------------------------------------------------------

    print()
    print(
        "Running ANN search..."
    )

    search_k = max(
        TOP_K_VALUES
    )

    distances, indices = (
        index.search(
            query_embeddings,
            search_k,
        )
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("=" * 75)
    print(
        "PHASE 4H RESULTS"
    )
    print("=" * 75)

    print()

    print(
        f"{'K':>8} "
        f"{'ANN Recall':>14} "
        f"{'Union Recall':>15} "
        f"{'Added TP':>12} "
        f"{'Complete':>12}"
    )

    print("-" * 70)

    corpus_ids = (
        corpus["entity_id"]
        .tolist()
    )

    for top_k in TOP_K_VALUES:

        ann_recovered = 0
        union_recovered = 0
        added_true_positives = 0
        complete_union = 0

        for query_index in range(
            len(validation)
        ):

            s1_id = validation.iloc[
                query_index
            ]["entity_id"]

            truth_set = truth[
                s1_id
            ]

            ann_candidates = {
                corpus_ids[index]
                for index in indices[
                    query_index,
                    :top_k,
                ]
                if index >= 0
            }

            phase4f = phase4f_sets[
                query_index
            ]

            union = (
                phase4f
                | ann_candidates
            )

            ann_hits = (
                truth_set
                & ann_candidates
            )

            union_hits = (
                truth_set
                & union
            )

            new_hits = (
                truth_set
                & ann_candidates
                - phase4f
            )

            ann_recovered += len(
                ann_hits
            )

            union_recovered += len(
                union_hits
            )

            added_true_positives += len(
                new_hits
            )

            if truth_set <= union:

                complete_union += 1

        print(
            f"{top_k:>8} "
            f"{ann_recovered / total_truth:>13.6%} "
            f"{union_recovered / total_truth:>14.6%} "
            f"{added_true_positives:>12,} "
            f"{complete_union:>12,}"
        )

    print()
    print("=" * 75)
    print(
        "PHASE 4H COMPLETE"
    )
    print("=" * 75)


if __name__ == "__main__":
    main()