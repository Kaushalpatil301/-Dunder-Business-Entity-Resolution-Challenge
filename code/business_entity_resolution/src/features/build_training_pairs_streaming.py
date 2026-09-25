from __future__ import annotations

import gc
import pickle
from pathlib import Path
from collections import defaultdict

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

NORM_DIR = PROJECT_ROOT / "data" / "normalized" / "train"
INDEX_DIR = PROJECT_ROOT / "data" / "blocking" / "indexes"
SPLIT_DIR = PROJECT_ROOT / "reports" / "splits"

OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

TRAIN_ENTITIES = 10_000

CHUNK_SIZE = 50_000

# Maximum negatives retained for each S1 entity.
MAX_NEGATIVES_PER_ENTITY = 50

# Keep up to this many positive matches.
MAX_POSITIVES_PER_ENTITY = 20


# ============================================================
# COLUMNS
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# HELPERS
# ============================================================

def load_pickle(path):
    print(f"Loading index: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def load_ground_truth():
    path = SPLIT_DIR / "train_ground_truth.tsv"

    print(f"Loading ground truth: {path}")

    gt = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    result = {}

    for _, row in gt.iterrows():

        ids = {
            x.strip()
            for x in row["matched_entity_ids"].split(",")
            if x.strip()
        }

        result[row["source1_entity_id"]] = ids

    return result


def load_train_s1_sample():
    """
    Load S1 entities strictly from the Phase 2 training split.
    """
    train_ids_path = (
        PROJECT_ROOT
        / "reports"
        / "splits"
        / "train_source1_ids.tsv"
    )

    train_ids_df = pd.read_csv(
        train_ids_path,
        sep="\t",
        header=None,
        names=["entity_id"],
        dtype=str,
    )

    train_ids = set(train_ids_df["entity_id"])

    rows = []

    s1_path = (
        PROJECT_ROOT
        / "data"
        / "normalized"
        / "train"
        / "train_source1_normalized.tsv"
    )

    for chunk in pd.read_csv(
        s1_path,
        sep="\t",
        dtype=str,
        chunksize=100_000,
    ):
        filtered = chunk[chunk["entity_id"].isin(train_ids)]

        if not filtered.empty:
            rows.append(filtered)

        if sum(len(x) for x in rows) >= TRAIN_ENTITIES:
            break

    if not rows:
        raise RuntimeError("No training S1 entities were found.")

    s1 = pd.concat(rows, ignore_index=True)

    # Deterministic sample
    s1 = (
        s1.sort_values("entity_id")
        .head(TRAIN_ENTITIES)
        .reset_index(drop=True)
    )

    return s1

# ============================================================
# BLOCKER HELPERS
# ============================================================

def get_postings(index_obj, key):

    """
    Supports the persisted index format used by Phase 4B.
    """

    if key not in index_obj:
        return []

    value = index_obj[key]

    if isinstance(value, list):
        return value

    if isinstance(value, set):
        return list(value)

    return list(value)


def collect_name_token_candidates(
    row,
    index,
):

    result = defaultdict(set)

    # Use all available name representations.
    representations = {
        "name_norm": row["name_norm"],
        "name_compact": row["name_compact"],
        "name_no_suffix": row["name_no_suffix"],
    }

    for representation, value in representations.items():

        tokens = str(value).split()

        for token in tokens:

            if not token:
                continue

            postings = get_postings(
                index,
                token,
            )

            for entity_id in postings:

                result[entity_id].add(
                    representation
                )

    return result


def collect_exact_address_candidates(
    row,
    index,
):

    result = defaultdict(set)

    address = str(
        row["address_norm"]
    ).strip()

    if not address:
        return result

    postings = get_postings(
        index,
        address,
    )

    for entity_id in postings:

        result[entity_id].add(
            "exact_address"
        )

    return result


def merge_provenance(
    destination,
    source,
):

    for entity_id, blockers in source.items():

        destination[entity_id].update(
            blockers
        )


# ============================================================
# BUILD BLOCKER CANDIDATES
# ============================================================

def build_candidates(
    s1_row,
    s2_name_index,
    s3_name_index,
    s2_address_index,
    s3_address_index,
):

    s2_candidates = defaultdict(set)
    s3_candidates = defaultdict(set)

    # --------------------------------------------------------
    # Name token blocking
    # --------------------------------------------------------

    merge_provenance(
        s2_candidates,
        collect_name_token_candidates(
            s1_row,
            s2_name_index,
        ),
    )

    merge_provenance(
        s3_candidates,
        collect_name_token_candidates(
            s1_row,
            s3_name_index,
        ),
    )

    # --------------------------------------------------------
    # Exact address blocking
    # --------------------------------------------------------

    merge_provenance(
        s2_candidates,
        collect_exact_address_candidates(
            s1_row,
            s2_address_index,
        ),
    )

    merge_provenance(
        s3_candidates,
        collect_exact_address_candidates(
            s1_row,
            s3_address_index,
        ),
    )

    return s2_candidates, s3_candidates


# ============================================================
# STREAM SOURCE AND KEEP REQUIRED IDS
# ============================================================

def retrieve_candidates_from_source(
    path,
    candidate_ids,
):

    candidate_ids = set(candidate_ids)

    if not candidate_ids:
        return {}

    found = {}

    print(
        f"Streaming {path.name} "
        f"for {len(candidate_ids):,} candidate IDs..."
    )

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=FEATURE_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):

        mask = chunk["entity_id"].isin(
            candidate_ids
        )

        selected = chunk.loc[mask]

        if not selected.empty:

            for _, row in selected.iterrows():

                entity_id = row["entity_id"]

                found[entity_id] = row.to_dict()

        del chunk

    return found


# ============================================================
# SAMPLE NEGATIVES
# ============================================================

def select_training_candidates(
    candidate_map,
    truth_ids,
):

    positive_ids = sorted(
        set(candidate_map)
        & truth_ids
    )

    negative_ids = sorted(
        set(candidate_map)
        - truth_ids
    )

    # Keep all positives up to the limit.
    positive_ids = positive_ids[
        :MAX_POSITIVES_PER_ENTITY
    ]

    # Deterministic hard-negative sampling.
    #
    # We currently retain the first N deterministic
    # negatives. Later, after feature computation, we'll
    # replace this with similarity-aware hard-negative mining.
    negative_ids = negative_ids[
        :MAX_NEGATIVES_PER_ENTITY
    ]

    return positive_ids, negative_ids


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 5B.1 — STREAMING TRAINING CANDIDATE BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load S1
    # --------------------------------------------------------

    s1 = load_train_s1_sample()
    s1_lookup = {
        row["entity_id"]: row.to_dict()
        for _, row in s1.iterrows()
    }

    # --------------------------------------------------------
    # 2. Ground truth
    # --------------------------------------------------------

    ground_truth = load_ground_truth()

    # --------------------------------------------------------
    # 3. Load blocker indexes
    # --------------------------------------------------------

    s2_name_obj = load_pickle(
        INDEX_DIR / "s2_name_token.pkl"
    )

    s3_name_obj = load_pickle(
        INDEX_DIR / "s3_name_token.pkl"
    )

    s2_address_obj = load_pickle(
        INDEX_DIR / "s2_address_exact.pkl"
    )

    s3_address_obj = load_pickle(
        INDEX_DIR / "s3_address_exact.pkl"
    )

    # Persisted token indexes use {"index": ...}.
    s2_name_index = s2_name_obj["index"]
    s3_name_index = s3_name_obj["index"]

    # Address indexes may be direct dictionaries or
    # wrapped dictionaries depending on the builder.
    if isinstance(s2_address_obj, dict) and "index" in s2_address_obj:
        s2_address_index = s2_address_obj["index"]
    else:
        s2_address_index = s2_address_obj

    if isinstance(s3_address_obj, dict) and "index" in s3_address_obj:
        s3_address_index = s3_address_obj["index"]
    else:
        s3_address_index = s3_address_obj

    # --------------------------------------------------------
    # 4. First pass:
    #    generate candidates and store only IDs/provenance
    # --------------------------------------------------------

    all_s2 = {}
    all_s3 = {}

    total_candidate_pairs = 0

    print("\nBuilding blocker candidate maps...")

    for i, (_, row) in enumerate(
        s1.iterrows(),
        start=1,
    ):

        s2_candidates, s3_candidates = build_candidates(
            row,
            s2_name_index,
            s3_name_index,
            s2_address_index,
            s3_address_index,
        )

        truth_ids = ground_truth.get(
            row["entity_id"],
            set(),
        )

        # ----------------------------------------------------
        # Sample candidates
        # ----------------------------------------------------

        s2_pos, s2_neg = select_training_candidates(
            s2_candidates,
            truth_ids,
        )

        s3_pos, s3_neg = select_training_candidates(
            s3_candidates,
            truth_ids,
        )

        selected_s2 = (
            s2_pos + s2_neg
        )

        selected_s3 = (
            s3_pos + s3_neg
        )

        # ----------------------------------------------------
        # Store required IDs and provenance
        # ----------------------------------------------------

        for entity_id in selected_s2:

            key = (
                row["entity_id"],
                entity_id,
                "S2",
            )

            all_s2[key] = {
                "source1_entity_id":
                    row["entity_id"],

                "candidate_entity_id":
                    entity_id,

                "candidate_source":
                    "S2",

                "label":
                    int(entity_id in truth_ids),

                "block_name_token":
                    int(
                        any(
                            x.startswith("name_")
                            for x in s2_candidates[
                                entity_id
                            ]
                        )
                    ),

                "block_exact_address":
                    int(
                        "exact_address"
                        in s2_candidates[
                            entity_id
                        ]
                    ),
            }

        for entity_id in selected_s3:

            key = (
                row["entity_id"],
                entity_id,
                "S3",
            )

            all_s3[key] = {
                "source1_entity_id":
                    row["entity_id"],

                "candidate_entity_id":
                    entity_id,

                "candidate_source":
                    "S3",

                "label":
                    int(entity_id in truth_ids),

                "block_name_token":
                    int(
                        any(
                            x.startswith("name_")
                            for x in s3_candidates[
                                entity_id
                            ]
                        )
                    ),

                "block_exact_address":
                    int(
                        "exact_address"
                        in s3_candidates[
                            entity_id
                        ]
                    ),
            }

        total_candidate_pairs += (
            len(selected_s2)
            + len(selected_s3)
        )

        if i % 500 == 0:

            print(
                f"Processed {i:,}/{len(s1):,} | "
                f"training pairs={total_candidate_pairs:,}"
            )

    print("\nCandidate ID collection complete.")

    # --------------------------------------------------------
    # 5. Retrieve S2 rows from disk
    # --------------------------------------------------------

    print("\nRetrieving selected S2 rows...")

    required_s2_ids = {
        key[1]
        for key in all_s2
    }

    s2_rows = retrieve_candidates_from_source(
        NORM_DIR / "train_source2_normalized.tsv",
        required_s2_ids,
    )

    print(
        f"S2 rows retrieved: {len(s2_rows):,}"
    )

    # --------------------------------------------------------
    # 6. Retrieve S3 rows from disk
    # --------------------------------------------------------

    print("\nRetrieving selected S3 rows...")

    required_s3_ids = {
        key[1]
        for key in all_s3
    }

    s3_rows = retrieve_candidates_from_source(
        NORM_DIR / "train_source3_normalized.tsv",
        required_s3_ids,
    )

    print(
        f"S3 rows retrieved: {len(s3_rows):,}"
    )

    # --------------------------------------------------------
    # 7. Build pair table
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / "training_pairs_sample.tsv"
    )

    rows = []

    # --------------------------------------------------------
    # S2
    # --------------------------------------------------------

    for key, metadata in all_s2.items():

        candidate_id = metadata[
            "candidate_entity_id"
        ]

        if candidate_id not in s2_rows:
            continue

        row = dict(metadata)

        # --------------------------------------------------------
        # Store S1 attributes
        # --------------------------------------------------------

        for column in FEATURE_COLUMNS:

            row[
                f"s1_{column}"
            ] = s1_lookup[
                row["source1_entity_id"]
            ].get(
                column,
                "",
            )

        candidate = s2_rows[
            candidate_id
        ]

        # Store candidate attributes.
        for column in FEATURE_COLUMNS:

            if column == "entity_id":
                continue

            row[
                f"candidate_{column}"
            ] = candidate.get(
                column,
                "",
            )

        rows.append(row)

    # --------------------------------------------------------
    # S3
    # --------------------------------------------------------

    for key, metadata in all_s3.items():

        candidate_id = metadata[
            "candidate_entity_id"
        ]

        if candidate_id not in s3_rows:
            continue

        row = dict(metadata)

        # --------------------------------------------------------
        # Store S1 attributes
        # --------------------------------------------------------

        for column in FEATURE_COLUMNS:

            row[
                f"s1_{column}"
            ] = s1_lookup[
                row["source1_entity_id"]
            ].get(
                column,
                "",
            )

        candidate = s3_rows[
            candidate_id
        ]

        for column in FEATURE_COLUMNS:

            if column == "entity_id":
                continue

            row[
                f"candidate_{column}"
            ] = candidate.get(
                column,
                "",
            )

        rows.append(row)

    output = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    output.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # 9. Statistics
    # --------------------------------------------------------

    positive_count = int(
        output["label"].sum()
    )

    negative_count = (
        len(output)
        - positive_count
    )

    print("\n" + "=" * 70)
    print("PHASE 5B.1 COMPLETE")
    print("=" * 70)

    print(
        f"Training pairs: {len(output):,}"
    )

    print(
        f"Positive pairs: {positive_count:,}"
    )

    print(
        f"Negative pairs: {negative_count:,}"
    )

    if len(output) > 0:

        print(
            f"Positive rate: "
            f"{positive_count / len(output):.4%}"
        )

    print(
        f"Output: {output_path}"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del s1
    del ground_truth
    del all_s2
    del all_s3
    del s2_rows
    del s3_rows
    del output

    gc.collect()


if __name__ == "__main__":
    main()