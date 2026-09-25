from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "lightgbm_validation_predictions.tsv"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "reports"
    / "splits"
    / "validation_ground_truth.tsv"
)


def load_ground_truth():

    df = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        dtype=str,
    )

    truth = {}

    for _, row in df.iterrows():

        s1 = row["source1_entity_id"]

        value = row["matched_entity_ids"]

        if pd.isna(value) or str(value).strip() == "":
            truth[s1] = set()
        else:
            truth[s1] = set(
                str(value).split("|")
            )

    return truth


def main():

    print("=" * 70)
    print("LIGHTGBM RANKING ANALYSIS")
    print("=" * 70)

    print("\nLoading predictions...")

    df = pd.read_csv(
        PREDICTIONS_FILE,
        sep="\t",
        dtype=str,
    )

    df["match_probability"] = pd.to_numeric(
        df["match_probability"]
    )

    print(
        f"Candidate rows: {len(df):,}"
    )

    truth = load_ground_truth()

    validation_ids = set(
        df["source1_entity_id"].unique()
    )

    truth = {
        s1: matches
        for s1, matches in truth.items()
        if s1 in validation_ids
    }

    print(
        f"Validation entities with candidates: "
        f"{len(validation_ids):,}"
    )

    print(
        f"Validation entities with truth: "
        f"{len(truth):,}"
    )

    # ---------------------------------------------------------
    # Rank candidates within each S1
    # ---------------------------------------------------------

    print("\nRanking candidates...")

    df = df.sort_values(
        [
            "source1_entity_id",
            "match_probability",
        ],
        ascending=[
            True,
            False,
        ],
    )

    df["rank"] = (
        df.groupby(
            "source1_entity_id"
        ).cumcount() + 1
    )

    # ---------------------------------------------------------
    # Analyze true-match ranks
    # ---------------------------------------------------------

    ranks = []

    reciprocal_ranks = []

    for s1, matches in truth.items():

        if not matches:
            continue

        group = df[
            df["source1_entity_id"] == s1
        ]

        rank_map = dict(
            zip(
                group["candidate_entity_id"],
                group["rank"],
            )
        )

        true_ranks = [
            rank_map[x]
            for x in matches
            if x in rank_map
        ]

        if not true_ranks:
            continue

        ranks.extend(true_ranks)

        reciprocal_ranks.append(
            1.0 / min(true_ranks)
        )

    if not ranks:

        print("\nNo true matches found in candidates.")

        return

    ranks = np.array(ranks)

    print("\n" + "=" * 70)
    print("TRUE MATCH RANK DISTRIBUTION")
    print("=" * 70)

    print(
        f"True matches present in candidates: "
        f"{len(ranks):,}"
    )

    print(
        f"Mean rank   : {ranks.mean():.2f}"
    )

    print(
        f"Median rank : {np.median(ranks):.2f}"
    )

    print(
        f"P95 rank    : {np.percentile(ranks, 95):.2f}"
    )

    print(
        f"Max rank    : {ranks.max():,}"
    )

    print("\nRecall by rank:")

    for k in [
        1,
        2,
        3,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
    ]:

        recall = (
            (ranks <= k).sum()
            / len(ranks)
        )

        print(
            f"Top-{k:<5}: "
            f"{recall:.6%}"
        )

    # ---------------------------------------------------------
    # Entity-level first-hit analysis
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ENTITY FIRST-TRUE-MATCH RANK")
    print("=" * 70)

    first_ranks = []

    for s1, matches in truth.items():

        if not matches:
            continue

        group = df[
            df["source1_entity_id"] == s1
        ]

        rank_map = dict(
            zip(
                group["candidate_entity_id"],
                group["rank"],
            )
        )

        true_ranks = [
            rank_map[x]
            for x in matches
            if x in rank_map
        ]

        if true_ranks:
            first_ranks.append(
                min(true_ranks)
            )

    first_ranks = np.array(first_ranks)

    print(
        f"Entities with >=1 true candidate: "
        f"{len(first_ranks):,}"
    )

    print(
        f"Median first true rank: "
        f"{np.median(first_ranks):.2f}"
    )

    print(
        f"P95 first true rank: "
        f"{np.percentile(first_ranks, 95):.2f}"
    )

    for k in [
        1,
        2,
        3,
        5,
        10,
        20,
        50,
        100,
    ]:

        recall = (
            (first_ranks <= k).sum()
            / len(first_ranks)
        )

        print(
            f"First true match Top-{k:<3}: "
            f"{recall:.6%}"
        )

    # ---------------------------------------------------------
    # Score distribution for positives
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRUE MATCH PROBABILITY")
    print("=" * 70)

    true_scores = []

    for s1, matches in truth.items():

        group = df[
            df["source1_entity_id"] == s1
        ]

        matched = group[
            group["candidate_entity_id"].isin(matches)
        ]

        true_scores.extend(
            matched["match_probability"].tolist()
        )

    true_scores = np.array(true_scores)

    print(
        f"True candidate scores: "
        f"{len(true_scores):,}"
    )

    print(
        f"Mean   : {true_scores.mean():.6f}"
    )

    print(
        f"Median : {np.median(true_scores):.6f}"
    )

    print(
        f"P10    : {np.percentile(true_scores, 10):.6f}"
    )

    print(
        f"P25    : {np.percentile(true_scores, 25):.6f}"
    )

    print(
        f"P50    : {np.percentile(true_scores, 50):.6f}"
    )

    print(
        f"P75    : {np.percentile(true_scores, 75):.6f}"
    )

    print(
        f"P90    : {np.percentile(true_scores, 90):.6f}"
    )

    # ---------------------------------------------------------
    # Top candidate correctness
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TOP CANDIDATE ANALYSIS")
    print("=" * 70)

    entities_with_truth = 0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    top10_correct = 0

    for s1, matches in truth.items():

        if not matches:
            continue

        group = df[
            df["source1_entity_id"] == s1
        ].sort_values(
            "match_probability",
            ascending=False,
        )

        if len(group) == 0:
            continue

        entities_with_truth += 1

        top1 = set(
            group.head(1)["candidate_entity_id"]
        )

        top3 = set(
            group.head(3)["candidate_entity_id"]
        )

        top5 = set(
            group.head(5)["candidate_entity_id"]
        )

        top10 = set(
            group.head(10)["candidate_entity_id"]
        )

        if top1 & matches:
            top1_correct += 1

        if top3 & matches:
            top3_correct += 1

        if top5 & matches:
            top5_correct += 1

        if top10 & matches:
            top10_correct += 1

    print(
        f"Entities with matches: "
        f"{entities_with_truth:,}"
    )

    print(
        f"Top-1 contains true match : "
        f"{top1_correct / entities_with_truth:.6%}"
    )

    print(
        f"Top-3 contains true match : "
        f"{top3_correct / entities_with_truth:.6%}"
    )

    print(
        f"Top-5 contains true match : "
        f"{top5_correct / entities_with_truth:.6%}"
    )

    print(
        f"Top-10 contains true match: "
        f"{top10_correct / entities_with_truth:.6%}"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()