from pathlib import Path
import csv
import time

from src.features.pair_features import build_pair_features


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

NORMALIZED_DIR = (
    PROJECT_ROOT
    / "data"
    / "normalized"
    / "train"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

S1_FILE = (
    NORMALIZED_DIR
    / "train_source1_normalized.tsv"
)

S2_DB = (
    PROJECT_ROOT
    / "data"
    / "lookup"
    / "s2.db"
)

S3_DB = (
    PROJECT_ROOT
    / "data"
    / "lookup"
    / "s3.db"
)

CANDIDATE_FILE = (
    FEATURE_DIR
    / "validation_candidates_sample.tsv"
)

OUTPUT_FILE = (
    FEATURE_DIR
    / "validation_pairs_features_sample.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 25_000


# ============================================================
# LOAD S1 RECORDS
# ============================================================

def load_s1_records():

    print("Loading S1 records...")

    start = time.time()

    rows = {}

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            rows[row["entity_id"]] = row

    print(
        f"Loaded {len(rows):,} S1 rows "
        f"in {time.time() - start:.1f}s"
    )

    return rows


# ============================================================
# SQLITE LOOKUP
# ============================================================

def fetch_candidate_rows(
    conn,
    candidate_ids,
):

    if not candidate_ids:
        return {}

    result = {}

    candidate_ids = list(
        candidate_ids
    )

    for start in range(
        0,
        len(candidate_ids),
        500,
    ):

        batch = candidate_ids[
            start:start + 500
        ]

        placeholders = ",".join(
            ["?"] * len(batch)
        )

        query = f"""
            SELECT *
            FROM entities
            WHERE entity_id IN ({placeholders})
        """

        cursor = conn.execute(
            query,
            batch,
        )

        columns = [
            description[0]
            for description in cursor.description
        ]

        for values in cursor.fetchall():

            row = dict(
                zip(
                    columns,
                    values,
                )
            )

            result[
                row["entity_id"]
            ] = row

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_total = time.time()

    print("=" * 70)
    print(
        "VALIDATION PAIR FEATURE GENERATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    s1_rows = load_s1_records()

    # --------------------------------------------------------
    # Read candidates
    # --------------------------------------------------------

    print(
        "\nReading validation candidates..."
    )

    candidates = []

    with open(
        CANDIDATE_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            candidates.append(row)

    print(
        f"Loaded {len(candidates):,} "
        f"candidate pairs"
    )

    # --------------------------------------------------------
    # Separate candidate IDs by source
    # --------------------------------------------------------

    s2_ids = set()
    s3_ids = set()

    for row in candidates:

        candidate_id = row[
            "candidate_entity_id"
        ]

        if row["candidate_source"] == "S2":
            s2_ids.add(candidate_id)

        elif row["candidate_source"] == "S3":
            s3_ids.add(candidate_id)

    print(
        f"S2 unique candidates : "
        f"{len(s2_ids):,}"
    )

    print(
        f"S3 unique candidates : "
        f"{len(s3_ids):,}"
    )

    # --------------------------------------------------------
    # Open SQLite databases
    # --------------------------------------------------------

    import sqlite3

    print(
        "\nOpening SQLite databases..."
    )

    conn_s2 = sqlite3.connect(
        f"file:{S2_DB}?mode=ro",
        uri=True,
    )

    conn_s3 = sqlite3.connect(
        f"file:{S3_DB}?mode=ro",
        uri=True,
    )

    # --------------------------------------------------------
    # Fetch candidate records
    # --------------------------------------------------------

    print(
        "\nFetching S2 candidate records..."
    )

    start = time.time()

    s2_rows = fetch_candidate_rows(
        conn_s2,
        s2_ids,
    )

    print(
        f"Fetched {len(s2_rows):,} S2 rows "
        f"in {time.time() - start:.1f}s"
    )

    print(
        "\nFetching S3 candidate records..."
    )

    start = time.time()

    s3_rows = fetch_candidate_rows(
        conn_s3,
        s3_ids,
    )

    print(
        f"Fetched {len(s3_rows):,} S3 rows "
        f"in {time.time() - start:.1f}s"
    )

    conn_s2.close()
    conn_s3.close()

    # --------------------------------------------------------
    # Determine output columns
    # --------------------------------------------------------

    feature_columns = None

    output_rows = []

    print(
        "\nBuilding pairwise features..."
    )

    processed = 0

    missing_s1 = 0
    missing_candidate = 0

    for start_idx in range(
        0,
        len(candidates),
        CHUNK_SIZE,
    ):

        chunk = candidates[
            start_idx:
            start_idx + CHUNK_SIZE
        ]

        for candidate in chunk:

            s1_id = candidate[
                "source1_entity_id"
            ]

            candidate_id = candidate[
                "candidate_entity_id"
            ]

            source = candidate[
                "candidate_source"
            ]

            s1 = s1_rows.get(
                s1_id
            )

            if s1 is None:

                missing_s1 += 1
                continue

            if source == "S2":

                candidate_row = (
                    s2_rows.get(
                        candidate_id
                    )
                )

            else:

                candidate_row = (
                    s3_rows.get(
                        candidate_id
                    )
                )

            if candidate_row is None:

                missing_candidate += 1
                continue

            # ------------------------------------------------
            # Build exact same pairwise features as training.
            # ------------------------------------------------

            features = build_pair_features(
                s1,
                candidate_row,
            )

            output = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": candidate_id,
                "candidate_source": source,
                "block_score": candidate[
                    "block_score"
                ],
            }

            output.update(
                features
            )

            if feature_columns is None:

                feature_columns = list(
                    features.keys()
                )

            output_rows.append(
                output
            )

            processed += 1

        print(
            f"  processed "
            f"{min(start_idx + CHUNK_SIZE, len(candidates)):,}"
            f"/{len(candidates):,}"
        )

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    print(
        "\nWriting validation feature file..."
    )

    fieldnames = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "block_score",
    ]

    if feature_columns:
        fieldnames.extend(
            feature_columns
        )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Input candidate pairs       : "
        f"{len(candidates):,}"
    )

    print(
        f"Feature rows written        : "
        f"{len(output_rows):,}"
    )

    print(
        f"Feature columns             : "
        f"{len(feature_columns or []):,}"
    )

    print(
        f"Missing S1 rows             : "
        f"{missing_s1:,}"
    )

    print(
        f"Missing candidate rows      : "
        f"{missing_candidate:,}"
    )

    print(
        f"Output                      : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Total runtime               : "
        f"{(time.time() - start_total) / 60:.2f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()