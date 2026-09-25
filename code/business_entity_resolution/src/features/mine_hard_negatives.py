from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "training_features_sample.tsv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "training_pairs_hard_negative.tsv"
)


# ============================================================
# CONFIG
# ============================================================

CHUNK_SIZE = 100_000

# Maximum hard negatives retained per S1 entity.
HARD_NEGATIVES_PER_ENTITY = 20

# Additional random/easier negatives per S1 entity.
RANDOM_NEGATIVES_PER_ENTITY = 10

RANDOM_STATE = 42


# ============================================================
# FEATURES USED ONLY FOR HARD-NEGATIVE RANKING
# ============================================================

DIFFICULTY_FEATURES = {
    "name_no_suffix_edit_similarity": 0.20,
    "name_jaccard": 0.15,
    "name_edit_similarity": 0.10,
    "name_token_intersection": 0.05,

    "address_edit_similarity": 0.15,
    "address_jaccard": 0.10,
    "address_numeric_jaccard": 0.10,
    "address_numeric_overlap": 0.10,
    "address_token_intersection": 0.05,
}


# ============================================================
# HELPERS
# ============================================================

def calculate_difficulty(df: pd.DataFrame) -> pd.Series:
    """
    Calculate a temporary similarity/difficulty score.

    This score is NOT the final entity-resolution model.
    It is used only to identify difficult negative examples.
    """

    score = np.zeros(len(df), dtype=np.float32)

    total_weight = sum(DIFFICULTY_FEATURES.values())

    for feature, weight in DIFFICULTY_FEATURES.items():
        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        ).fillna(0.0)

        score += (
            values.to_numpy(dtype=np.float32)
            * weight
        )

    return pd.Series(
        score / total_weight,
        index=df.index,
        name="_difficulty_score",
    )


def select_entity_negatives(
    group: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Select hard + random negatives for one S1 entity.
    """

    positives = group[group["label"] == 1]

    negatives = group[group["label"] == 0].copy()

    # Always preserve every available positive.
    selected_parts = [positives]

    if negatives.empty:
        return pd.concat(
            selected_parts,
            ignore_index=True
        )

    # --------------------------------------------------------
    # Hard negatives
    # --------------------------------------------------------

    negatives = negatives.sort_values(
        "_difficulty_score",
        ascending=False,
        kind="stable",
    )

    hard_count = min(
        HARD_NEGATIVES_PER_ENTITY,
        len(negatives),
    )

    hard = negatives.iloc[:hard_count].copy()

    selected_negative_ids = set(
        hard["candidate_entity_id"].astype(str)
    )

    # --------------------------------------------------------
    # Random negatives
    # --------------------------------------------------------

    remaining = negatives.iloc[hard_count:].copy()

    random_count = min(
        RANDOM_NEGATIVES_PER_ENTITY,
        len(remaining),
    )

    if random_count > 0:
        random_positions = rng.choice(
            len(remaining),
            size=random_count,
            replace=False,
        )

        random_part = remaining.iloc[
            random_positions
        ].copy()

        # Make sure we don't duplicate a hard negative.
        random_part = random_part[
            ~random_part["candidate_entity_id"]
            .astype(str)
            .isin(selected_negative_ids)
        ]

    else:
        random_part = remaining.iloc[0:0].copy()

    selected_parts.extend(
        [hard, random_part]
    )

    return pd.concat(
        selected_parts,
        ignore_index=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PHASE 5E — HARD-NEGATIVE MINING")
    print("=" * 80)

    print(f"Input : {FEATURE_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {FEATURE_PATH}"
        )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    # --------------------------------------------------------
    # Read the feature file
    # --------------------------------------------------------

    print()
    print("Loading feature data...")

    df = pd.read_csv(
        FEATURE_PATH,
        sep="\t",
        dtype={
            "source1_entity_id": str,
            "candidate_entity_id": str,
            "candidate_source": str,
            "label": np.int8,
        },
    )

    print(
        f"Input rows: {len(df):,}"
    )

    print(
        f"Input positives: "
        f"{int(df['label'].sum()):,}"
    )

    print(
        f"Input negatives: "
        f"{int((df['label'] == 0).sum()):,}"
    )

    print(
        f"Unique S1 entities: "
        f"{df['source1_entity_id'].nunique():,}"
    )

    # --------------------------------------------------------
    # Difficulty score
    # --------------------------------------------------------

    print()
    print("Calculating negative difficulty scores...")

    df["_difficulty_score"] = calculate_difficulty(df)

    # --------------------------------------------------------
    # Entity-level sampling
    # --------------------------------------------------------

    print()
    print("Selecting hard negatives per S1 entity...")

    selected = []

    entity_count = 0

    for entity_id, group in df.groupby(
        "source1_entity_id",
        sort=False,
    ):

        result = select_entity_negatives(
            group,
            rng,
        )

        selected.append(result)

        entity_count += 1

        if entity_count % 1000 == 0:
            print(
                f"Processed entities: "
                f"{entity_count:,}"
            )

    output = pd.concat(
        selected,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Remove temporary ranking feature
    # --------------------------------------------------------

    output = output.drop(
        columns=["_difficulty_score"]
    )

    # --------------------------------------------------------
    # Deterministic ordering
    # --------------------------------------------------------

    output = output.sort_values(
        [
            "source1_entity_id",
            "label",
            "candidate_entity_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    positives = int(
        output["label"].sum()
    )

    negatives = (
        len(output)
        - positives
    )

    entities = (
        output["source1_entity_id"]
        .nunique()
    )

    negatives_per_entity = (
        output[output["label"] == 0]
        .groupby("source1_entity_id")
        .size()
    )

    print()
    print("=" * 80)
    print("PHASE 5E COMPLETE")
    print("=" * 80)

    print(
        f"Output rows: {len(output):,}"
    )

    print(
        f"Positive pairs: {positives:,}"
    )

    print(
        f"Negative pairs: {negatives:,}"
    )

    print(
        f"Positive rate: "
        f"{positives / len(output):.4%}"
    )

    print(
        f"Unique S1 entities: "
        f"{entities:,}"
    )

    if not negatives_per_entity.empty:
        print()
        print("Negative pairs per entity:")
        print(
            f"  Mean  : "
            f"{negatives_per_entity.mean():.2f}"
        )
        print(
            f"  Median: "
            f"{negatives_per_entity.median():.2f}"
        )
        print(
            f"  Max   : "
            f"{negatives_per_entity.max():.0f}"
        )

    print()
    print(
        f"Output: {OUTPUT_PATH}"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del df
    del output
    del selected

    gc.collect()


if __name__ == "__main__":
    main()
