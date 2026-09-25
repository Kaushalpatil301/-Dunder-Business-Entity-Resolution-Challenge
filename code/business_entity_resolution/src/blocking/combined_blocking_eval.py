from __future__ import annotations

import pickle
import re
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 100_000

ADDRESS_THRESHOLDS = [
    100,
    500,
    1_000,
    2_500,
]

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
# HELPERS
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
# TRUTH
# ============================================================

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

    for row in df.itertuples(index=False):

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
# S1 VALIDATION DATA
# ============================================================

def load_s1(
    path: Path,
) -> pd.DataFrame:

    columns = [
        "entity_id",
        "name_norm",
        "address_norm",
    ]

    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=columns,
    )


# ============================================================
# NAME TOKEN INDEX
# ============================================================

def load_name_token_indexes():

    root = index_root()

    s2_payload = load_pickle(
        root / "s2_name_token.pkl"
    )

    s3_payload = load_pickle(
        root / "s3_name_token.pkl"
    )

    return {
        "s2": s2_payload["index"],
        "s3": s3_payload["index"],
    }


# ============================================================
# ADDRESS / NUMERIC INDEXES
# ============================================================

def load_address_indexes():

    root = index_root()

    exact = {
        "s2": load_pickle(
            root / "s2_address_exact.pkl"
        ),
        "s3": load_pickle(
            root / "s3_address_exact.pkl"
        ),
    }

    numeric = {
        "s2": load_pickle(
            root / "s2_numeric.pkl"
        ),
        "s3": load_pickle(
            root / "s3_numeric.pkl"
        ),
    }

    address_tokens = {}

    for threshold in ADDRESS_THRESHOLDS:

        address_tokens[threshold] = {
            "s2": load_pickle(
                root
                / f"s2_address_token_{threshold}.pkl"
            ),
            "s3": load_pickle(
                root
                / f"s3_address_token_{threshold}.pkl"
            ),
        }

    return (
        exact,
        numeric,
        address_tokens,
    )


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    row,
    name_token_indexes,
    exact_address_indexes,
    numeric_indexes,
    address_token_indexes,
    address_threshold,
):

    candidates = set()

    # ========================================================
    # NAME TOKEN BLOCKING
    # ========================================================

    name_tokens = tokenize(
        row.name_norm
    )

    for token in name_tokens:

        candidates.update(
            name_token_indexes["s2"].get(
                token,
                [],
            )
        )

        candidates.update(
            name_token_indexes["s3"].get(
                token,
                [],
            )
        )

    # ========================================================
    # EXACT ADDRESS BLOCKING
    # ========================================================

    address = row.address_norm

    if address:

        candidates.update(
            exact_address_indexes["s2"].get(
                address,
                [],
            )
        )

        candidates.update(
            exact_address_indexes["s3"].get(
                address,
                [],
            )
        )

    # ========================================================
    # ADDRESS TOKEN BLOCKING
    # ========================================================

    address_tokens = tokenize(
        address
    )

    for token in address_tokens:

        candidates.update(
            address_token_indexes[
                address_threshold
            ]["s2"].get(
                token,
                [],
            )
        )

        candidates.update(
            address_token_indexes[
                address_threshold
            ]["s3"].get(
                token,
                [],
            )
        )

    # ========================================================
    # NUMERIC ADDRESS BLOCKING
    # ========================================================

    numbers = extract_numbers(
        address
    )

    for number in numbers:

        candidates.update(
            numeric_indexes["s2"].get(
                number,
                [],
            )
        )

        candidates.update(
            numeric_indexes["s3"].get(
                number,
                [],
            )
        )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "PHASE 4F - COMBINED BLOCKER UNION"
    )
    print("=" * 75)

    root = normalized_root()

    s1_path = (
        root
        / "train_source1_normalized.tsv"
    )

    truth_path = (
        splits_root()
        / "validation_ground_truth.tsv"
    )

    # ========================================================
    # LOAD INDEXES
    # ========================================================

    print()
    print("Loading name-token indexes...")

    name_token_indexes = (
        load_name_token_indexes()
    )

    print("Name-token indexes loaded.")

    print()
    print("Loading address/numeric indexes...")

    (
        exact_address_indexes,
        numeric_indexes,
        address_token_indexes,
    ) = load_address_indexes()

    print("Address/numeric indexes loaded.")

    # ========================================================
    # LOAD VALIDATION DATA
    # ========================================================

    print()
    print("Loading validation data...")

    s1_df = load_s1(
        s1_path
    )

    truth = load_truth(
        truth_path
    )

    print(
        f"S1 validation rows: "
        f"{len(s1_df):,}"
    )

    # ========================================================
    # RESULT STORAGE
    # ========================================================

    matched_entities = 0
    total_true_links = 0

    results = {}

    for threshold in ADDRESS_THRESHOLDS:

        results[threshold] = {
            "true_recovered": 0,
            "complete": 0,
            "candidate_counts": [],
        }

    # ========================================================
    # EVALUATION
    # ========================================================

    print()
    print("=" * 75)
    print("GENERATING COMBINED CANDIDATES")
    print("=" * 75)

    for row in s1_df.itertuples(
        index=False
    ):

        s1_id = row.entity_id

        true_matches = truth.get(
            s1_id
        )

        # Only evaluate entities having
        # at least one true match.
        if not true_matches:
            continue

        matched_entities += 1

        total_true_links += len(
            true_matches
        )

        for threshold in ADDRESS_THRESHOLDS:

            candidates = generate_candidates(
                row=row,
                name_token_indexes=name_token_indexes,
                exact_address_indexes=(
                    exact_address_indexes
                ),
                numeric_indexes=(
                    numeric_indexes
                ),
                address_token_indexes=(
                    address_token_indexes
                ),
                address_threshold=threshold,
            )

            recovered = (
                true_matches
                & candidates
            )

            results[
                threshold
            ][
                "true_recovered"
            ] += len(
                recovered
            )

            if recovered == true_matches:

                results[
                    threshold
                ][
                    "complete"
                ] += 1

            results[
                threshold
            ][
                "candidate_counts"
            ].append(
                len(candidates)
            )

        if matched_entities % 25_000 == 0:

            print(
                f"Processed "
                f"{matched_entities:,} "
                f"matched entities"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 75)
    print("PHASE 4F RESULTS")
    print("=" * 75)

    print()
    print(
        f"Validation entities with matches: "
        f"{matched_entities:,}"
    )

    print(
        f"Total true links: "
        f"{total_true_links:,}"
    )

    print()

    print(
        f"{'Addr DF':>10} "
        f"{'Recall':>14} "
        f"{'Complete':>14} "
        f"{'Mean cand.':>14} "
        f"{'Median':>12} "
        f"{'P95':>12} "
        f"{'Max':>12}"
    )

    print("-" * 95)

    for threshold in ADDRESS_THRESHOLDS:

        data = results[
            threshold
        ]

        recall = (
            data["true_recovered"]
            / total_true_links
            if total_true_links
            else 0.0
        )

        series = pd.Series(
            data["candidate_counts"]
        )

        mean_candidates = (
            series.mean()
            if len(series)
            else 0.0
        )

        median_candidates = (
            series.median()
            if len(series)
            else 0.0
        )

        p95_candidates = (
            series.quantile(0.95)
            if len(series)
            else 0.0
        )

        max_candidates = (
            series.max()
            if len(series)
            else 0
        )

        print(
            f"{threshold:>10,} "
            f"{recall:>13.6%} "
            f"{data['complete']:>14,} "
            f"{mean_candidates:>14.2f} "
            f"{median_candidates:>12.2f} "
            f"{p95_candidates:>12.2f} "
            f"{max_candidates:>12,}"
        )

    print()
    print("=" * 75)
    print("PHASE 4F COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()