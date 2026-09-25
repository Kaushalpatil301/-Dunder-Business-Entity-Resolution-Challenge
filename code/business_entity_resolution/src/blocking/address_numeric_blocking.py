from __future__ import annotations

import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 100_000

MIN_TOKEN_LENGTH = 3

# Address-token DF thresholds to test.
TOKEN_THRESHOLDS = [
    100,
    250,
    500,
    1_000,
    2_500,
    5_000,
    10_000,
]

# Numeric tokens with extremely high DF are not useful.
MAX_NUMERIC_DF = 10_000

NUMBER_PATTERN = re.compile(r"\d+")


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
    path = (
        project_root()
        / "data"
        / "blocking"
        / "indexes"
    )
    path.mkdir(
        parents=True,
        exist_ok=True,
    )
    return path


def splits_root() -> Path:
    return (
        project_root()
        / "reports"
        / "splits"
    )


def tokenize_address(address: str) -> list[str]:
    if not address:
        return []

    tokens = re.findall(
        r"[a-z0-9]+",
        str(address).casefold(),
    )

    return [
        token
        for token in tokens
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def extract_numbers(address: str) -> list[str]:
    if not address:
        return []

    return NUMBER_PATTERN.findall(
        str(address)
    )


def build_address_token_df(
    path: Path,
) -> tuple[Counter, int]:

    token_df = Counter()
    total_rows = 0

    print(
        f"Scanning address tokens: {path.name}"
    )

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "address_norm",
        ],
        chunksize=CHUNK_SIZE,
    ):

        for address in chunk["address_norm"]:

            tokens = set(
                tokenize_address(address)
            )

            token_df.update(tokens)

        total_rows += len(chunk)

        print(
            f"  scanned {total_rows:,} rows"
        )

    return token_df, total_rows


def build_numeric_df(
    path: Path,
) -> tuple[Counter, int]:

    numeric_df = Counter()
    total_rows = 0

    print(
        f"Scanning numeric tokens: {path.name}"
    )

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "address_norm",
        ],
        chunksize=CHUNK_SIZE,
    ):

        for address in chunk["address_norm"]:

            numbers = set(
                extract_numbers(address)
            )

            numeric_df.update(numbers)

        total_rows += len(chunk)

        print(
            f"  scanned {total_rows:,} rows"
        )

    return numeric_df, total_rows


def build_address_token_index(
    path: Path,
    output_path: Path,
    df_threshold: int,
) -> None:

    token_index = defaultdict(list)

    total_rows = 0
    indexed_rows = 0
    postings = 0

    print()
    print(
        f"Building address-token index "
        f"for DF <= {df_threshold:,}"
    )

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "address_norm",
        ],
        chunksize=CHUNK_SIZE,
    ):

        for row in chunk.itertuples(
            index=False
        ):

            tokens = set(
                tokenize_address(
                    row.address_norm
                )
            )

            useful_tokens = [
                token
                for token in tokens
                if df_cache.get(
                    token,
                    df_threshold + 1,
                ) <= df_threshold
            ]

            if useful_tokens:
                indexed_rows += 1

            for token in useful_tokens:
                token_index[token].append(
                    row.entity_id
                )
                postings += 1

        total_rows += len(chunk)

        print(
            f"  indexed {total_rows:,} rows"
        )

    token_index = dict(token_index)

    with open(
        output_path,
        "wb",
    ) as f:
        pickle.dump(
            token_index,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"  indexed tokens: "
        f"{len(token_index):,}"
    )

    print(
        f"  indexed rows: "
        f"{indexed_rows:,}"
    )

    print(
        f"  postings: "
        f"{postings:,}"
    )

    print(
        f"  saved: "
        f"{output_path}"
    )


def build_numeric_index(
    path: Path,
    output_path: Path,
) -> None:

    numeric_index = defaultdict(list)

    total_rows = 0
    indexed_rows = 0
    postings = 0

    print()
    print(
        "Building numeric-token index"
    )

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "address_norm",
        ],
        chunksize=CHUNK_SIZE,
    ):

        for row in chunk.itertuples(
            index=False
        ):

            numbers = set(
                extract_numbers(
                    row.address_norm
                )
            )

            useful_numbers = [
                number
                for number in numbers
                if numeric_df_cache.get(
                    number,
                    MAX_NUMERIC_DF + 1,
                ) <= MAX_NUMERIC_DF
            ]

            if useful_numbers:
                indexed_rows += 1

            for number in useful_numbers:
                numeric_index[number].append(
                    row.entity_id
                )
                postings += 1

        total_rows += len(chunk)

        print(
            f"  indexed {total_rows:,} rows"
        )

    numeric_index = dict(numeric_index)

    with open(
        output_path,
        "wb",
    ) as f:
        pickle.dump(
            numeric_index,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"  indexed numbers: "
        f"{len(numeric_index):,}"
    )

    print(
        f"  indexed rows: "
        f"{indexed_rows:,}"
    )

    print(
        f"  postings: "
        f"{postings:,}"
    )

    print(
        f"  saved: "
        f"{output_path}"
    )


def build_exact_address_index(
    path: Path,
    output_path: Path,
) -> None:

    address_index = defaultdict(list)

    total_rows = 0

    print()
    print(
        f"Building exact-address index: "
        f"{path.name}"
    )

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "address_norm",
        ],
        chunksize=CHUNK_SIZE,
    ):

        for row in chunk.itertuples(
            index=False
        ):

            address = row.address_norm.strip()

            if address:
                address_index[address].append(
                    row.entity_id
                )

        total_rows += len(chunk)

        print(
            f"  indexed {total_rows:,} rows"
        )

    with open(
        output_path,
        "wb",
    ) as f:
        pickle.dump(
            dict(address_index),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"  unique addresses: "
        f"{len(address_index):,}"
    )

    print(
        f"  saved: "
        f"{output_path}"
    )


def load_truth(
    path: Path,
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
        matched = row.matched_entity_ids

        if not matched:
            truth[s1_id] = set()
        else:
            truth[s1_id] = {
                x.strip()
                for x in matched.split(",")
                if x.strip()
            }

    return truth


def load_s1_validation(
    path: Path,
) -> pd.DataFrame:

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "address_norm",
        ],
    )


def evaluate_candidates(
    s1_df: pd.DataFrame,
    truth: dict[str, set[str]],
    exact_indexes,
    token_indexes,
    numeric_indexes,
) -> None:

    total_true = 0

    exact_recovered = 0
    token_recovered = {
        threshold: 0
        for threshold in TOKEN_THRESHOLDS
    }
    numeric_recovered = 0

    combined_recovered = {
        threshold: 0
        for threshold in TOKEN_THRESHOLDS
    }

    exact_candidates = []
    token_candidates = {
        threshold: []
        for threshold in TOKEN_THRESHOLDS
    }
    numeric_candidates = []
    combined_candidates = {
        threshold: []
        for threshold in TOKEN_THRESHOLDS
    }

    exact_complete = 0

    token_complete = {
        threshold: 0
        for threshold in TOKEN_THRESHOLDS
    }

    combined_complete = {
        threshold: 0
        for threshold in TOKEN_THRESHOLDS
    }

    matched_entities = 0

    print()
    print("=" * 70)
    print("EVALUATING ADDRESS + NUMERIC BLOCKERS")
    print("=" * 70)

    for row in s1_df.itertuples(
        index=False
    ):

        s1_id = row.entity_id

        true_matches = truth.get(
            s1_id,
            set(),
        )

        if not true_matches:
            continue

        matched_entities += 1

        total_true += len(
            true_matches
        )

        address = row.address_norm

        # -----------------------------------------
        # Exact address
        # -----------------------------------------

        exact_set = set()

        if address:
            for index in exact_indexes:
                exact_set.update(
                    index.get(
                        address,
                        [],
                    )
                )

        exact_candidates.append(
            len(exact_set)
        )

        exact_hits = (
            true_matches
            & exact_set
        )

        exact_recovered += len(
            exact_hits
        )

        if (
            exact_hits
            == true_matches
        ):
            exact_complete += 1

        # -----------------------------------------
        # Numeric
        # -----------------------------------------

        numeric_set = set()

        for number in extract_numbers(
            address
        ):
            for index in numeric_indexes:
                numeric_set.update(
                    index.get(
                        number,
                        [],
                    )
                )

        numeric_candidates.append(
            len(numeric_set)
        )

        numeric_recovered += len(
            true_matches
            & numeric_set
        )

        # -----------------------------------------
        # Address token thresholds
        # -----------------------------------------

        threshold_sets = {
            threshold: set()
            for threshold in TOKEN_THRESHOLDS
        }

        tokens = set(
            tokenize_address(address)
        )

        for token in tokens:

            for threshold, index in token_indexes.items():

                if (
                    df_cache.get(
                        token,
                        threshold + 1,
                    )
                    <= threshold
                ):
                    threshold_sets[
                        threshold
                    ].update(
                        index.get(
                            token,
                            [],
                        )
                    )

        for threshold in TOKEN_THRESHOLDS:

            candidates = threshold_sets[
                threshold
            ]

            token_candidates[
                threshold
            ].append(
                len(candidates)
            )

            hits = (
                true_matches
                & candidates
            )

            token_recovered[
                threshold
            ] += len(hits)

            if hits == true_matches:
                token_complete[
                    threshold
                ] += 1

            # -------------------------------------
            # Combined address + numeric
            # -------------------------------------

            combined = (
                candidates
                | exact_set
                | numeric_set
            )

            combined_candidates[
                threshold
            ].append(
                len(combined)
            )

            combined_hits = (
                true_matches
                & combined
            )

            combined_recovered[
                threshold
            ] += len(combined_hits)

            if (
                combined_hits
                == true_matches
            ):
                combined_complete[
                    threshold
                ] += 1

        if matched_entities % 50_000 == 0:
            print(
                f"Processed "
                f"{matched_entities:,} "
                f"matched S1 entities"
            )

    print()
    print("=" * 70)
    print("ADDRESS BLOCKING RESULTS")
    print("=" * 70)

    print()
    print(
        f"Validation entities with matches: "
        f"{matched_entities:,}"
    )

    print(
        f"Total true links: "
        f"{total_true:,}"
    )

    print()

    exact_recall = (
        exact_recovered / total_true
        if total_true
        else 0
    )

    print(
        "Exact normalized address"
    )

    print(
        f"  Recall: "
        f"{exact_recall:.6%}"
    )

    print(
        f"  Complete entities: "
        f"{exact_complete:,}"
    )

    print(
        f"  Mean candidates: "
        f"{sum(exact_candidates) / len(exact_candidates):.2f}"
    )

    print(
        f"  Median candidates: "
        f"{pd.Series(exact_candidates).median():.2f}"
    )

    print()

    print(
        "Numeric-token blocker"
    )

    numeric_recall = (
        numeric_recovered / total_true
        if total_true
        else 0
    )

    print(
        f"  Recall: "
        f"{numeric_recall:.6%}"
    )

    print(
        f"  Mean candidates: "
        f"{sum(numeric_candidates) / len(numeric_candidates):.2f}"
    )

    print(
        f"  Median candidates: "
        f"{pd.Series(numeric_candidates).median():.2f}"
    )

    print()

    print(
        f"{'DF cap':>10} "
        f"{'Token Recall':>16} "
        f"{'Combined Recall':>18} "
        f"{'Token Mean':>14} "
        f"{'Combined Mean':>16} "
        f"{'Complete':>12}"
    )

    print("-" * 92)

    for threshold in TOKEN_THRESHOLDS:

        token_recall = (
            token_recovered[threshold]
            / total_true
        )

        combined_recall = (
            combined_recovered[threshold]
            / total_true
        )

        token_mean = (
            sum(
                token_candidates[threshold]
            )
            / len(
                token_candidates[threshold]
            )
        )

        combined_mean = (
            sum(
                combined_candidates[threshold]
            )
            / len(
                combined_candidates[threshold]
            )
        )

        print(
            f"{threshold:>10,} "
            f"{token_recall:>15.6%} "
            f"{combined_recall:>17.6%} "
            f"{token_mean:>14.2f} "
            f"{combined_mean:>16.2f} "
            f"{combined_complete[threshold]:>12,}"
        )


def main() -> None:

    global df_cache
    global numeric_df_cache

    print("=" * 70)
    print("PHASE 4E - ADDRESS + NUMERIC BLOCKING")
    print("=" * 70)

    root = normalized_root()
    indexes = index_root()

    s2_path = (
        root
        / "train_source2_normalized.tsv"
    )

    s3_path = (
        root
        / "train_source3_normalized.tsv"
    )

    s1_path = (
        root
        / "train_source1_normalized.tsv"
    )

    truth_path = (
        splits_root()
        / "validation_ground_truth.tsv"
    )

    # --------------------------------------------------
    # Build DF statistics
    # --------------------------------------------------

    start = time.time()

    s2_address_df, _ = (
        build_address_token_df(
            s2_path
        )
    )

    s3_address_df, _ = (
        build_address_token_df(
            s3_path
        )
    )

    combined_address_df = (
        s2_address_df
        + s3_address_df
    )

    print()
    print(
        f"Combined address tokens: "
        f"{len(combined_address_df):,}"
    )

    print(
        f"Address DF build time: "
        f"{time.time() - start:.2f}s"
    )

    # Make globally available to the helper.
    df_cache = combined_address_df

    # --------------------------------------------------
    # Numeric DF
    # --------------------------------------------------

    start = time.time()

    s2_numeric_df, _ = (
        build_numeric_df(
            s2_path
        )
    )

    s3_numeric_df, _ = (
        build_numeric_df(
            s3_path
        )
    )

    combined_numeric_df = (
        s2_numeric_df
        + s3_numeric_df
    )

    numeric_df_cache = (
        combined_numeric_df
    )

    print()
    print(
        f"Combined numeric tokens: "
        f"{len(combined_numeric_df):,}"
    )

    print(
        f"Numeric DF build time: "
        f"{time.time() - start:.2f}s"
    )

    # --------------------------------------------------
    # Build exact address indexes
    # --------------------------------------------------

    s2_exact_path = (
        indexes
        / "s2_address_exact.pkl"
    )

    s3_exact_path = (
        indexes
        / "s3_address_exact.pkl"
    )

    build_exact_address_index(
        s2_path,
        s2_exact_path,
    )

    build_exact_address_index(
        s3_path,
        s3_exact_path,
    )

    # --------------------------------------------------
    # Build address-token indexes
    # --------------------------------------------------

    token_indexes = {}

    for threshold in TOKEN_THRESHOLDS:

        print()
        print(
            "=" * 70
        )

        s2_token_path = (
            indexes
            / f"s2_address_token_{threshold}.pkl"
        )

        s3_token_path = (
            indexes
            / f"s3_address_token_{threshold}.pkl"
        )

        build_address_token_index(
            s2_path,
            s2_token_path,
            threshold,
        )

        build_address_token_index(
            s3_path,
            s3_token_path,
            threshold,
        )

        with open(
            s2_token_path,
            "rb",
        ) as f:
            s2_index = pickle.load(f)

        with open(
            s3_token_path,
            "rb",
        ) as f:
            s3_index = pickle.load(f)

        combined = defaultdict(list)

        for token, ids in s2_index.items():
            combined[token].extend(ids)

        for token, ids in s3_index.items():
            combined[token].extend(ids)

        token_indexes[threshold] = dict(
            combined
        )

        # We don't need to keep the individual
        # S2/S3 indexes after combining them.

        del s2_index
        del s3_index

    # --------------------------------------------------
    # Build numeric indexes
    # --------------------------------------------------

    s2_numeric_path = (
        indexes
        / "s2_numeric.pkl"
    )

    s3_numeric_path = (
        indexes
        / "s3_numeric.pkl"
    )

    build_numeric_index(
        s2_path,
        s2_numeric_path,
    )

    build_numeric_index(
        s3_path,
        s3_numeric_path,
    )

    with open(
        s2_numeric_path,
        "rb",
    ) as f:
        s2_numeric_index = pickle.load(f)

    with open(
        s3_numeric_path,
        "rb",
    ) as f:
        s3_numeric_index = pickle.load(f)

    numeric_indexes = [
        s2_numeric_index,
        s3_numeric_index,
    ]

    # --------------------------------------------------
    # Load validation
    # --------------------------------------------------

    print()
    print(
        "Loading validation data..."
    )

    s1_df = load_s1_validation(
        s1_path
    )

    truth = load_truth(
        truth_path
    )

    exact_indexes = []

    with open(
        s2_exact_path,
        "rb",
    ) as f:
        exact_indexes.append(
            pickle.load(f)
        )

    with open(
        s3_exact_path,
        "rb",
    ) as f:
        exact_indexes.append(
            pickle.load(f)
        )

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    evaluate_candidates(
        s1_df=s1_df,
        truth=truth,
        exact_indexes=exact_indexes,
        token_indexes=token_indexes,
        numeric_indexes=numeric_indexes,
    )

    print()
    print("=" * 70)
    print("PHASE 4E COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()