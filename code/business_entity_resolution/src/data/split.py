from pathlib import Path
import pandas as pd
import numpy as np


RANDOM_SEED = 42

# Validation proportion
VALIDATION_FRACTION = 0.20


def load_ground_truth(path):
    """
    Load train_ground_truth.tsv.

    Returns a DataFrame with:
        source1_entity_id
        matched_entity_ids
        match_count
    """

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")

    df["match_count"] = df["matched_entity_ids"].apply(
        lambda x: 0 if not x.strip()
        else len([
            item.strip()
            for item in x.split(",")
            if item.strip()
        ])
    )

    return df


def create_entity_split(
    ground_truth,
    validation_fraction=VALIDATION_FRACTION,
    random_seed=RANDOM_SEED
):
    """
    Split Source 1 entities into train and validation sets.

    The split happens at the Source 1 entity level.
    All matches belonging to one Source 1 entity remain
    in the same split.
    """

    rng = np.random.default_rng(random_seed)

    source1_ids = ground_truth["source1_entity_id"].to_numpy()

    shuffled_ids = source1_ids.copy()
    rng.shuffle(shuffled_ids)

    validation_size = int(
        len(shuffled_ids) * validation_fraction
    )

    validation_ids = set(
        shuffled_ids[:validation_size]
    )

    train_mask = ~ground_truth["source1_entity_id"].isin(
        validation_ids
    )

    validation_mask = ground_truth["source1_entity_id"].isin(
        validation_ids
    )

    train_ground_truth = ground_truth.loc[
        train_mask
    ].copy()

    validation_ground_truth = ground_truth.loc[
        validation_mask
    ].copy()

    return train_ground_truth, validation_ground_truth


def summarize_split(name, df):
    """
    Print useful information about a split.
    """

    total = len(df)

    singleton_count = (
        (df["match_count"] == 0)
        .sum()
    )

    multi_match_count = (
        (df["match_count"] > 0)
        .sum()
    )

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(f"Source 1 entities: {total:,}")

    if total > 0:
        print(
            f"Singletons: "
            f"{singleton_count:,} "
            f"({singleton_count / total:.4%})"
        )

        print(
            f"Entities with matches: "
            f"{multi_match_count:,} "
            f"({multi_match_count / total:.4%})"
        )

    print(
        f"Positive links: "
        f"{df['match_count'].sum():,}"
    )

    print(
        f"Average matches per entity: "
        f"{df['match_count'].mean():.4f}"
    )

    print(
        f"Maximum matches per entity: "
        f"{df['match_count'].max()}"
    )


def main():

    # Resolve project root:
    #
    # src/data/split.py
    #       ↓
    # business_entity_resolution
    #       ↓
    # code
    #       ↓
    # project root

    project_root = Path(__file__).resolve().parents[4]

    ground_truth_path = (
        project_root
        / "dataset"
        / "train"
        / "train_ground_truth.tsv"
    )

    split_dir = (
        project_root
        / "reports"
        / "splits"
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Loading ground truth:\n"
        f"{ground_truth_path}"
    )

    ground_truth = load_ground_truth(
        ground_truth_path
    )

    print(
        f"\nTotal Source 1 entities: "
        f"{len(ground_truth):,}"
    )

    train_gt, validation_gt = create_entity_split(
        ground_truth
    )

    summarize_split(
        "TRAIN SPLIT",
        train_gt
    )

    summarize_split(
        "VALIDATION SPLIT",
        validation_gt
    )

    # Check that every S1 appears exactly once.
    train_ids = set(
        train_gt["source1_entity_id"]
    )

    validation_ids = set(
        validation_gt["source1_entity_id"]
    )

    overlap = train_ids & validation_ids

    if overlap:
        raise RuntimeError(
            f"Data leakage detected: "
            f"{len(overlap)} Source 1 IDs appear "
            f"in both train and validation."
        )

    all_ids = train_ids | validation_ids

    original_ids = set(
        ground_truth["source1_entity_id"]
    )

    if all_ids != original_ids:
        raise RuntimeError(
            "Some Source 1 entities were lost "
            "during the split."
        )

    print()
    print("=" * 60)
    print("LEAKAGE CHECK")
    print("=" * 60)

    print("Train/validation overlap: 0")
    print("All Source 1 entities preserved: YES")

    # Save the split IDs.
    train_ids_df = pd.DataFrame({
        "source1_entity_id":
            sorted(train_ids)
    })

    validation_ids_df = pd.DataFrame({
        "source1_entity_id":
            sorted(validation_ids)
    })

    train_ids_path = (
        split_dir
        / "train_source1_ids.tsv"
    )

    validation_ids_path = (
        split_dir
        / "validation_source1_ids.tsv"
    )

    train_ids_df.to_csv(
        train_ids_path,
        sep="\t",
        index=False
    )

    validation_ids_df.to_csv(
        validation_ids_path,
        sep="\t",
        index=False
    )

    # Save complete ground-truth subsets too.
    train_gt_path = (
        split_dir
        / "train_ground_truth.tsv"
    )

    validation_gt_path = (
        split_dir
        / "validation_ground_truth.tsv"
    )

    train_gt.drop(
        columns=["match_count"]
    ).to_csv(
        train_gt_path,
        sep="\t",
        index=False
    )

    validation_gt.drop(
        columns=["match_count"]
    ).to_csv(
        validation_gt_path,
        sep="\t",
        index=False
    )

    print()
    print("Files written:")
    print(train_ids_path)
    print(validation_ids_path)
    print(train_gt_path)
    print(validation_gt_path)

    print()
    print("PHASE 2 COMPLETE")


if __name__ == "__main__":
    main()