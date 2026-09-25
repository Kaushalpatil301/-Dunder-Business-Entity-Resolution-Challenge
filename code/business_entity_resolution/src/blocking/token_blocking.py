from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 100_000

MIN_TOKEN_LENGTH = 3
MAX_TOKEN_DF_RATIO = 0.005


def tokenize_name(value: str) -> list[str]:
    """
    Extract normalized name tokens.

    The normalized dataset already contains name_tokens,
    so we only split the existing representation here.
    """
    if not value:
        return []

    return [
        token
        for token in str(value).split()
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def build_token_statistics(
    path: Path,
) -> tuple[int, Counter]:
    """
    First pass over a source file.

    Calculates:
        - total rows
        - document frequency of each name token

    A token is counted at most once per entity.
    """

    print()
    print("=" * 70)
    print(f"BUILDING TOKEN STATISTICS")
    print(path)
    print("=" * 70)

    token_df = Counter()
    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["name_tokens"],
        chunksize=CHUNK_SIZE,
    ):

        for value in chunk["name_tokens"]:

            tokens = set(tokenize_name(value))

            for token in tokens:
                token_df[token] += 1

        total_rows += len(chunk)

        print(
            f"Statistics rows: {total_rows:,}"
        )

    print(
        f"Total rows: {total_rows:,}"
    )

    print(
        f"Unique tokens: {len(token_df):,}"
    )

    return total_rows, token_df


def select_rare_tokens(
    total_rows: int,
    token_df: Counter,
) -> set[str]:
    """
    Keep tokens that are sufficiently rare to be useful for blocking.
    """

    max_df = total_rows * MAX_TOKEN_DF_RATIO

    rare_tokens = {
        token
        for token, frequency in token_df.items()
        if frequency <= max_df
    }

    print()
    print(
        f"Maximum allowed document frequency: "
        f"{max_df:,.0f}"
    )

    print(
        f"Rare/informative tokens: "
        f"{len(rare_tokens):,}"
    )

    return rare_tokens


def build_token_index(
    path: Path,
    rare_tokens: set[str],
) -> dict[str, list[str]]:
    """
    Second pass.

    Build:

        token -> entity IDs
    """

    print()
    print("=" * 70)
    print("BUILDING PERSISTENT TOKEN INDEX")
    print(path)
    print("=" * 70)

    index = defaultdict(list)

    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "name_tokens"],
        chunksize=CHUNK_SIZE,
    ):

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            tokens = set(
                tokenize_name(row.name_tokens)
            )

            for token in tokens:

                if token in rare_tokens:
                    index[token].append(entity_id)

        total_rows += len(chunk)

        print(
            f"Index rows: {total_rows:,}"
        )

    result = dict(index)

    print(
        f"Indexed tokens: {len(result):,}"
    )

    total_postings = sum(
        len(ids)
        for ids in result.values()
    )

    print(
        f"Total token postings: "
        f"{total_postings:,}"
    )

    return result


def save_index(
    index: dict[str, list[str]],
    rare_tokens: set[str],
    total_rows: int,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "index": index,
        "rare_tokens": rare_tokens,
        "total_rows": total_rows,
        "min_token_length": MIN_TOKEN_LENGTH,
        "max_token_df_ratio": MAX_TOKEN_DF_RATIO,
    }

    with open(output_path, "wb") as file:
        pickle.dump(
            payload,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(
        f"Saved index: {output_path}"
    )


def load_index(
    path: Path,
) -> dict:

    with open(path, "rb") as file:
        return pickle.load(file)


def build_source_index(
    input_path: Path,
    output_path: Path,
) -> None:

    total_rows, token_df = build_token_statistics(
        input_path
    )

    rare_tokens = select_rare_tokens(
        total_rows,
        token_df,
    )

    index = build_token_index(
        input_path,
        rare_tokens,
    )

    save_index(
        index,
        rare_tokens,
        total_rows,
        output_path,
    )


def generate_token_candidates(
    s1_path: Path,
    s2_index: dict,
    s3_index: dict,
    truth: dict[str, set[str]],
) -> None:

    print()
    print("=" * 70)
    print("PHASE 4C - SELECTIVE RARE-TOKEN BLOCKING")
    print("=" * 70)

    s2_token_index = s2_index["index"]
    s3_token_index = s3_index["index"]

    # Test several posting-list/document-frequency caps.
    thresholds = [
        100,
        250,
        500,
        1_000,
        2_500,
        5_000,
        10_000,
        25_000,
    ]

    # Statistics for each threshold.
    stats = {
        threshold: {
            "recovered_links": 0,
            "complete_entities": 0,
            "candidate_counts": [],
        }
        for threshold in thresholds
    }

    total_rows = 0
    matched_entities = 0
    total_truth_links = 0

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id", "name_tokens"],
        chunksize=CHUNK_SIZE,
    ):

        for row in chunk.itertuples(index=False):

            s1_id = row.entity_id

            true_matches = truth.get(s1_id)

            if true_matches is None:
                continue

            matched_entities += 1
            total_truth_links += len(true_matches)

            tokens = set(
                tokenize_name(row.name_tokens)
            )

            # One candidate set per threshold.
            candidates_by_threshold = {
                threshold: set()
                for threshold in thresholds
            }

            for token in tokens:

                s2_postings = s2_token_index.get(
                    token,
                    [],
                )

                s3_postings = s3_token_index.get(
                    token,
                    [],
                )

                # IMPORTANT:
                # The threshold is applied to the combined
                # posting frequency of the token.
                token_df = (
                    len(s2_postings)
                    + len(s3_postings)
                )

                for threshold in thresholds:

                    if token_df > threshold:
                        continue

                    candidates = candidates_by_threshold[
                        threshold
                    ]

                    candidates.update(
                        s2_postings
                    )

                    candidates.update(
                        s3_postings
                    )

            # Evaluate every threshold immediately.
            for threshold in thresholds:

                candidates = candidates_by_threshold[
                    threshold
                ]

                stats[threshold][
                    "candidate_counts"
                ].append(
                    len(candidates)
                )

                recovered = (
                    true_matches & candidates
                )

                stats[threshold][
                    "recovered_links"
                ] += len(recovered)

                if recovered == true_matches:
                    stats[threshold][
                        "complete_entities"
                    ] += 1

        total_rows += len(chunk)

        print(
            f"Processed S1 rows: "
            f"{total_rows:,}"
        )

    print()
    print("=" * 70)
    print("PHASE 4C RESULTS")
    print("=" * 70)

    print(
        f"Validation entities with matches: "
        f"{matched_entities:,}"
    )

    print(
        f"Total true links: "
        f"{total_truth_links:,}"
    )

    print()

    print(
        f"{'DF Cap':>10} "
        f"{'Recall':>12} "
        f"{'Complete':>12} "
        f"{'Mean':>12} "
        f"{'Median':>12} "
        f"{'P95':>12} "
        f"{'Max':>12}"
    )

    print("-" * 88)

    for threshold in thresholds:

        recovered_links = stats[
            threshold
        ]["recovered_links"]

        complete_entities = stats[
            threshold
        ]["complete_entities"]

        candidate_counts = stats[
            threshold
        ]["candidate_counts"]

        series = pd.Series(
            candidate_counts
        )

        recall = (
            recovered_links
            / total_truth_links
            if total_truth_links
            else 0.0
        )

        print(
            f"{threshold:>10,} "
            f"{recall:>11.6%} "
            f"{complete_entities:>12,} "
            f"{series.mean():>12.2f} "
            f"{series.median():>12.2f} "
            f"{series.quantile(0.95):>12.2f} "
            f"{series.max():>12,}"
        )

    print()
    print(
        "Interpretation:"
    )
    print(
        "Lower DF caps produce smaller candidate sets "
        "but may reduce recall."
    )
    print(
        "Higher DF caps improve recall but increase "
        "downstream matching cost."
    )


def evaluate_candidates(
    candidates: dict[str, set[str]],
    truth: dict[str, set[str]],
) -> None:

    total_truth_links = sum(
        len(matches)
        for matches in truth.values()
    )

    recovered_links = 0
    complete_entities = 0

    candidate_counts = []

    for s1_id, true_matches in truth.items():

        generated = candidates.get(
            s1_id,
            set(),
        )

        candidate_counts.append(
            len(generated)
        )

        recovered = (
            true_matches & generated
        )

        recovered_links += len(recovered)

        if recovered == true_matches:
            complete_entities += 1

    series = pd.Series(candidate_counts)

    recall = (
        recovered_links / total_truth_links
        if total_truth_links
        else 0.0
    )

    print()
    print("=" * 70)
    print("RARE-TOKEN BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"Total true links: "
        f"{total_truth_links:,}"
    )

    print(
        f"Recovered true links: "
        f"{recovered_links:,}"
    )

    print(
        f"Candidate recall: "
        f"{recall:.6%}"
    )

    print(
        f"Entities with all true links recovered: "
        f"{complete_entities:,}"
    )

    print()
    print("Candidate counts:")

    print(
        f"Mean:   {series.mean():.2f}"
    )

    print(
        f"Median: {series.median():.2f}"
    )

    print(
        f"P95:    {series.quantile(0.95):.2f}"
    )

    print(
        f"Max:    {series.max():,}"
    )


def load_truth(path: Path) -> dict[str, set[str]]:

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = defaultdict(set)

    for row in df.itertuples(index=False):

        s1_id = row.source1_entity_id
        matches = row.matched_entity_ids

        if not matches:
            continue

        for match_id in matches.split(","):

            match_id = match_id.strip()

            if match_id:
                truth[s1_id].add(
                    match_id
                )

    return dict(truth)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build",
        action="store_true",
        help="Build persistent S2/S3 token indexes.",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate rare-token blocking.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]

    normalized_root = (
        project_root
        / "data"
        / "normalized"
        / "train"
    )

    index_root = (
        project_root
        / "data"
        / "blocking"
        / "indexes"
    )

    if args.build:

        build_source_index(
            normalized_root
            / "train_source2_normalized.tsv",
            index_root
            / "s2_name_token.pkl",
        )

        build_source_index(
            normalized_root
            / "train_source3_normalized.tsv",
            index_root
            / "s3_name_token.pkl",
        )

    if args.evaluate:

        s2_index = load_index(
            index_root
            / "s2_name_token.pkl"
        )

        s3_index = load_index(
            index_root
            / "s3_name_token.pkl"
        )

        truth = load_truth(
            project_root
            / "reports"
            / "splits"
            / "validation_ground_truth.tsv"
        )

        generate_token_candidates(
            normalized_root
            / "train_source1_normalized.tsv",
            s2_index,
            s3_index,
            truth,
        )

    if not args.build and not args.evaluate:
        parser.print_help()


if __name__ == "__main__":
    main()