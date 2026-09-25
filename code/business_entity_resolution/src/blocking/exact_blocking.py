from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 100_000


BLOCKING_COLUMNS = [
    "name_norm",
    "name_compact",
    "name_no_suffix",
]


def load_validation_ground_truth(path: Path) -> dict[str, set[str]]:
    """
    Load validation ground truth into:

        S1 entity_id -> set of true S2/S3 IDs
    """

    print(f"Loading validation ground truth: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    truth = defaultdict(set)

    for row in df.itertuples(index=False):
        s1_id = row.source1_entity_id
        match_list = row.matched_entity_ids

        if not match_list:
            continue

        for match_id in match_list.split(","):
            match_id = match_id.strip()

            if match_id:
                truth[s1_id].add(match_id)

    print(f"Validation entities with matches: {len(truth):,}")

    return dict(truth)

def build_index(
    path: Path,
    source_prefix: str,
) -> dict[str, dict[str, list[str]]]:
    """
    Build three exact-match indexes.

    Returns:

        {
            "name_norm": {
                key: [entity_id, ...]
            },
            ...
        }

    Empty keys are ignored.
    """

    print()
    print("=" * 70)
    print(f"BUILDING INDEX: {source_prefix}")
    print(path)
    print("=" * 70)

    indexes = {
        column: defaultdict(list)
        for column in BLOCKING_COLUMNS
    }

    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id"] + BLOCKING_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id

            for column_index, column in enumerate(BLOCKING_COLUMNS, start=1):
                key = row[column_index]

                if key:
                    indexes[column][key].append(entity_id)

        total_rows += len(chunk)

        print(f"Indexed rows: {total_rows:,}")

    result = {
        column: dict(index)
        for column, index in indexes.items()
    }

    for column in BLOCKING_COLUMNS:
        print(
            f"{column}: "
            f"{len(result[column]):,} unique keys"
        )

    return result


def generate_candidates(
    s1_path: Path,
    indexes: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    """
    Generate the union of candidates from all exact blockers.

    Returns:

        all_candidates:
            S1 -> set(candidate IDs)

        blocker_candidates:
            S1 -> blocker -> set(candidate IDs)
    """

    print()
    print("=" * 70)
    print("GENERATING VALIDATION CANDIDATES")
    print("=" * 70)

    all_candidates = defaultdict(set)

    blocker_candidates = defaultdict(
        lambda: defaultdict(set)
    )

    total_rows = 0

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=["entity_id"] + BLOCKING_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):

        for row in chunk.itertuples(index=False):

            s1_id = row.entity_id

            for column_index, column in enumerate(
                BLOCKING_COLUMNS,
                start=1,
            ):

                key = row[column_index]

                if not key:
                    continue

                matches = indexes[column].get(key, [])

                for candidate_id in matches:
                    all_candidates[s1_id].add(candidate_id)
                    blocker_candidates[s1_id][column].add(
                        candidate_id
                    )

        total_rows += len(chunk)

        print(f"Processed S1 rows: {total_rows:,}")

    return (
        dict(all_candidates),
        {
            s1_id: dict(blockers)
            for s1_id, blockers in blocker_candidates.items()
        },
    )


def evaluate_candidates(
    truth: dict[str, set[str]],
    candidates: dict[str, set[str]],
    blocker_candidates: dict[str, dict[str, set[str]]],
) -> dict:
    """
    Evaluate candidate recall.
    """

    total_truth_links = sum(
        len(matches)
        for matches in truth.values()
    )

    recovered_links = 0

    candidate_counts = []

    blocker_recovered = {
        column: 0
        for column in BLOCKING_COLUMNS
    }

    entities_with_truth = len(truth)

    entities_with_all_truth_recovered = 0

    for s1_id, true_matches in truth.items():

        generated = candidates.get(s1_id, set())

        candidate_counts.append(len(generated))

        recovered = true_matches & generated

        recovered_links += len(recovered)

        if recovered == true_matches:
            entities_with_all_truth_recovered += 1

        for column in BLOCKING_COLUMNS:

            blocker_set = (
                blocker_candidates
                .get(s1_id, {})
                .get(column, set())
            )

            blocker_recovered[column] += len(
                true_matches & blocker_set
            )

    candidate_series = pd.Series(candidate_counts)

    overall_recall = (
        recovered_links / total_truth_links
        if total_truth_links
        else 0.0
    )

    print()
    print("=" * 70)
    print("CANDIDATE RECALL RESULTS")
    print("=" * 70)

    print(
        f"Validation entities with matches: "
        f"{entities_with_truth:,}"
    )

    print(
        f"Total true links: "
        f"{total_truth_links:,}"
    )

    print(
        f"Recovered true links: "
        f"{recovered_links:,}"
    )

    print(
        f"Overall candidate recall: "
        f"{overall_recall:.6%}"
    )

    print(
        f"Entities with all true links recovered: "
        f"{entities_with_all_truth_recovered:,}"
    )

    print()
    print("Blocker-level recall:")

    for column in BLOCKING_COLUMNS:

        recall = (
            blocker_recovered[column]
            / total_truth_links
            if total_truth_links
            else 0.0
        )

        print(
            f"  {column:20s} "
            f"{blocker_recovered[column]:,} links "
            f"({recall:.6%})"
        )

    print()
    print("Candidate counts per matched S1 entity:")

    print(
        f"  Mean:   {candidate_series.mean():.2f}"
    )

    print(
        f"  Median: {candidate_series.median():.2f}"
    )

    print(
        f"  P95:    {candidate_series.quantile(0.95):.2f}"
    )

    print(
        f"  Max:    {candidate_series.max():,}"
    )

    print()
    print("=" * 70)

    return {
        "total_truth_links": total_truth_links,
        "recovered_links": recovered_links,
        "overall_recall": overall_recall,
        "entities_with_all_truth_recovered": (
            entities_with_all_truth_recovered
        ),
        "mean_candidates": candidate_series.mean(),
        "median_candidates": candidate_series.median(),
        "p95_candidates": candidate_series.quantile(0.95),
        "max_candidates": candidate_series.max(),
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Phase 4A exact blocking"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process the first N validation S1 rows.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]

    normalized_root = (
        project_root
        / "data"
        / "normalized"
        / "train"
    )

    split_root = (
        project_root
        / "reports"
        / "splits"
    )

    validation_s1 = (
        normalized_root
        / "train_source1_normalized.tsv"
    )

    s2 = (
        normalized_root
        / "train_source2_normalized.tsv"
    )

    s3 = (
        normalized_root
        / "train_source3_normalized.tsv"
    )

    ground_truth = (
        split_root
        / "validation_ground_truth.tsv"
    )

    truth = load_validation_ground_truth(
        ground_truth
    )

    print()
    print("Building S2 index...")

    s2_index = build_index(
        s2,
        "S2",
    )

    print()
    print("Building S3 index...")

    s3_index = build_index(
        s3,
        "S3",
    )

    combined_indexes = {}

    for column in BLOCKING_COLUMNS:

        merged = defaultdict(list)

        for key, ids in s2_index[column].items():
            merged[key].extend(ids)

        for key, ids in s3_index[column].items():
            merged[key].extend(ids)

        combined_indexes[column] = dict(merged)

    candidates, blocker_candidates = generate_candidates(
        validation_s1,
        combined_indexes,
    )

    evaluate_candidates(
        truth,
        candidates,
        blocker_candidates,
    )

    print()
    print("PHASE 4A COMPLETE")


if __name__ == "__main__":
    main()