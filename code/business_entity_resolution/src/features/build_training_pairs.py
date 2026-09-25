from __future__ import annotations

import gc
import pickle
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

NORM_DIR = PROJECT_ROOT / "data" / "normalized" / "train"
INDEX_DIR = PROJECT_ROOT / "data" / "blocking" / "indexes"
SPLIT_DIR = PROJECT_ROOT / "reports" / "splits"

OUTPUT_DIR = PROJECT_ROOT / "data" / "features"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

# Number of training S1 entities to use initially.
# We deliberately start small.
TRAIN_ENTITIES = 10_000

# Maximum negative candidates retained per S1 entity.
MAX_NEGATIVES_PER_ENTITY = 50

# Keep a reasonable number of positives.
MAX_POSITIVES_PER_ENTITY = 20

CHUNK_SIZE = 50_000


# ============================================================
# LOAD INDEX
# ============================================================

def load_pickle(path):
    print(f"Loading index: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# LOAD NORMALIZED DATA
# ============================================================

def load_source(path, nrows=None):

    usecols = [
        "entity_id",
        "name_norm",
        "name_compact",
        "name_no_suffix",
        "name_sorted",
        "name_translit",
        "address_norm",
        "address_compact",
        "address_sorted",
        "address_translit",
        "country_norm",
    ]

    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        nrows=nrows,
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth():

    path = (
        SPLIT_DIR
        / "train_ground_truth.tsv"
    )

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    return {
        row["source1_entity_id"]:
            set(
                x.strip()
                for x in row["matched_entity_ids"].split(",")
                if x.strip()
            )
        for _, row in gt.iterrows()
    }


# ============================================================
# TOKEN INDEX CANDIDATES
# ============================================================

def token_candidates(
    entity,
    token_index,
):

    candidates = set()

    tokens = (
        str(entity["name_norm"])
        .split()
    )

    for token in tokens:

        postings = token_index.get(
            token,
            [],
        )

        candidates.update(
            postings
        )

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 5B — TRAINING CANDIDATE SAMPLER")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load training S1
    # --------------------------------------------------------

    s1_path = (
        NORM_DIR
        / "train_source1_normalized.tsv"
    )

    s1 = load_source(s1_path)

    print(
        f"S1 records loaded: {len(s1):,}"
    )

    # Deterministic sample for the first experiment.
    s1 = (
        s1
        .sort_values("entity_id")
        .head(TRAIN_ENTITIES)
        .copy()
    )

    print(
        f"S1 entities sampled: {len(s1):,}"
    )

    # --------------------------------------------------------
    # 2. Load S2/S3
    # --------------------------------------------------------

    s2_path = (
        NORM_DIR
        / "train_source2_normalized.tsv"
    )

    s3_path = (
        NORM_DIR
        / "train_source3_normalized.tsv"
    )

    s2 = load_source(s2_path)
    s3 = load_source(s3_path)

    s2["source"] = "S2"
    s3["source"] = "S3"

    candidates_df = pd.concat(
        [s2, s3],
        ignore_index=True,
    )

    candidates_df = candidates_df.set_index(
        "entity_id",
        drop=False,
    )

    print(
        f"S2 records: {len(s2):,}"
    )

    print(
        f"S3 records: {len(s3):,}"
    )

    # --------------------------------------------------------
    # 3. Load indexes
    # --------------------------------------------------------

    s2_index_path = (
        INDEX_DIR
        / "s2_name_token.pkl"
    )

    s3_index_path = (
        INDEX_DIR
        / "s3_name_token.pkl"
    )

    s2_index_obj = load_pickle(
        s2_index_path
    )

    s3_index_obj = load_pickle(
        s3_index_path
    )

    # Your token indexes contain:
    # {
    #   "index": {...}
    # }

    s2_index = s2_index_obj["index"]
    s3_index = s3_index_obj["index"]

    # --------------------------------------------------------
    # 4. Ground truth
    # --------------------------------------------------------

    print("Loading ground truth...")

    ground_truth = load_ground_truth()

    # --------------------------------------------------------
    # 5. Generate pairs
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "training_pairs_sample.tsv"
    )

    first_write = True

    total_pairs = 0
    total_positive = 0
    total_negative = 0

    print("\nGenerating candidate pairs...")

    for row_number, (_, s1_row) in enumerate(
        s1.iterrows(),
        start=1,
    ):

        s1_id = s1_row["entity_id"]

        truth_ids = ground_truth.get(
            s1_id,
            set(),
        )

        # ----------------------------------------------------
        # Generate S2 candidates
        # ----------------------------------------------------

        s2_candidates = token_candidates(
            s1_row,
            s2_index,
        )

        # ----------------------------------------------------
        # Generate S3 candidates
        # ----------------------------------------------------

        s3_candidates = token_candidates(
            s1_row,
            s3_index,
        )

        candidate_ids = (
            s2_candidates
            | s3_candidates
        )

        if not candidate_ids:
            continue

        # ----------------------------------------------------
        # Separate positives / negatives
        # ----------------------------------------------------

        positive_ids = (
            candidate_ids
            & truth_ids
        )

        negative_ids = (
            candidate_ids
            - truth_ids
        )

        # ----------------------------------------------------
        # Keep all positives up to limit
        # ----------------------------------------------------

        positive_ids = sorted(
            positive_ids
        )[:MAX_POSITIVES_PER_ENTITY]

        # ----------------------------------------------------
        # Deterministic negative sampling
        # ----------------------------------------------------

        negative_ids = sorted(
            negative_ids
        )[:MAX_NEGATIVES_PER_ENTITY]

        selected = []

        for candidate_id in positive_ids:

            selected.append(
                (
                    s1_id,
                    candidate_id,
                    1,
                )
            )

        for candidate_id in negative_ids:

            selected.append(
                (
                    s1_id,
                    candidate_id,
                    0,
                )
            )

        if not selected:
            continue

        # ----------------------------------------------------
        # Build output
        # ----------------------------------------------------

        rows = []

        for (
            source1_id,
            candidate_id,
            label,
        ) in selected:

            candidate = candidates_df.loc[
                candidate_id
            ]

            rows.append(
                {
                    "source1_entity_id":
                        source1_id,

                    "candidate_entity_id":
                        candidate_id,

                    "candidate_source":
                        candidate["source"],

                    "label":
                        label,
                }
            )

        output_df = pd.DataFrame(rows)

        output_df.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False

        pair_count = len(output_df)

        total_pairs += pair_count
        total_positive += int(
            output_df["label"].sum()
        )
        total_negative += (
            pair_count
            - int(output_df["label"].sum())
        )

        if row_number % 500 == 0:

            print(
                f"Processed {row_number:,} | "
                f"pairs={total_pairs:,} | "
                f"positive={total_positive:,} | "
                f"negative={total_negative:,}"
            )

    print("\n" + "=" * 70)
    print("TRAINING PAIR SAMPLING COMPLETE")
    print("=" * 70)

    print(
        f"Total pairs: {total_pairs:,}"
    )

    print(
        f"Positive pairs: {total_positive:,}"
    )

    print(
        f"Negative pairs: {total_negative:,}"
    )

    print(
        f"Output: {output_path}"
    )

    del s1
    del s2
    del s3
    del candidates_df

    gc.collect()


if __name__ == "__main__":
    main()