from pathlib import Path
import csv
import pickle
import sqlite3
import statistics
import time

from rapidfuzz.fuzz import ratio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

NORMALIZED_DIR = PROJECT_ROOT / "data" / "normalized" / "train"
BLOCKING_DIR = PROJECT_ROOT / "data" / "blocking" / "indexes"
LOOKUP_DIR = PROJECT_ROOT / "data" / "lookup"

SPLIT_DIR = PROJECT_ROOT / "reports" / "splits"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"

S1_FILE = NORMALIZED_DIR / "train_source1_normalized.tsv"

S2_DB = LOOKUP_DIR / "s2.db"
S3_DB = LOOKUP_DIR / "s3.db"

S2_NAME_INDEX = BLOCKING_DIR / "s2_name_token.pkl"
S3_NAME_INDEX = BLOCKING_DIR / "s3_name_token.pkl"

S2_ADDRESS_EXACT_INDEX = BLOCKING_DIR / "s2_address_exact.pkl"
S3_ADDRESS_EXACT_INDEX = BLOCKING_DIR / "s3_address_exact.pkl"

S2_ADDRESS_TOKEN_INDEX = BLOCKING_DIR / "s2_address_token_1000.pkl"
S3_ADDRESS_TOKEN_INDEX = BLOCKING_DIR / "s3_address_token_1000.pkl"

S2_NUMERIC_INDEX = BLOCKING_DIR / "s2_numeric.pkl"
S3_NUMERIC_INDEX = BLOCKING_DIR / "s3_numeric.pkl"

VALIDATION_IDS_FILE = SPLIT_DIR / "validation_source1_ids.tsv"
VALIDATION_TRUTH_FILE = SPLIT_DIR / "validation_ground_truth.tsv"

OUTPUT_FILE = FEATURE_DIR / "validation_candidates_sample.tsv"


# ============================================================
# PILOT SETTINGS
# ============================================================

VALIDATION_ENTITY_LIMIT = 500

# Keep this at 2000 for the diagnostic.
TOP_K_PER_SOURCE = 10000

ADDRESS_DF_CAP = 1000

SQL_BATCH_SIZE = 500

MIN_TOKEN_LENGTH = 3


# ============================================================
# INDEX LOADING
# ============================================================

def load_pickle_index(path):
    """
    Load a blocking index.

    Some indexes are stored directly as:
        {key: postings}

    Others are wrapped as:
        {
            "index": {key: postings},
            ...
        }

    We only unwrap "index" when its value is actually a dict.
    """

    print(f"Loading index: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired blocking index does not exist:\n{path}"
        )

    with open(path, "rb") as f:
        obj = pickle.load(f)

    print(f"  top-level type: {type(obj).__name__}")

    if not isinstance(obj, dict):
        raise TypeError(
            f"Expected dictionary index in {path}, "
            f"got {type(obj).__name__}"
        )

    # Handle wrapped indexes such as:
    # {"index": {...}, ...}
    nested_index = obj.get("index")

    if isinstance(nested_index, dict):
        print(
            f"  using nested 'index': "
            f"{len(nested_index):,} entries"
        )
        return nested_index

    # Direct dictionary index.
    print(f"  entries: {len(obj):,}")

    return obj


# ============================================================
# VALIDATION DATA
# ============================================================

def load_validation_ids():

    ids = []

    with open(
        VALIDATION_IDS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            entity_id = row.get("entity_id")

            if entity_id is None:
                entity_id = row.get(
                    "source1_entity_id"
                )

            if entity_id:
                ids.append(entity_id)

            if len(ids) >= VALIDATION_ENTITY_LIMIT:
                break

    return ids


def load_validation_truth():

    truth = {}

    with open(
        VALIDATION_TRUTH_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:

            s1_id = row["source1_entity_id"]

            raw_ids = row[
                "matched_entity_ids"
            ].strip()

            if not raw_ids:

                truth[s1_id] = set()

            else:

                truth[s1_id] = {
                    x.strip()
                    for x in raw_ids.split(",")
                    if x.strip()
                }

    return truth


def load_validation_s1(validation_ids):

    wanted = set(validation_ids)

    rows = {}

    print("\nLoading validation S1 records...")

    start = time.time()

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

            entity_id = row["entity_id"]

            if entity_id in wanted:

                rows[entity_id] = row

                if len(rows) == len(wanted):
                    break

    print(
        f"Loaded {len(rows):,}/{len(wanted):,} S1 records "
        f"in {time.time() - start:.1f}s"
    )

    return rows


# ============================================================
# TOKEN / POSTING HELPERS
# ============================================================

def safe_tokens(value):

    if not value:
        return []

    return [
        token
        for token in value.split()
        if len(token) >= MIN_TOKEN_LENGTH
    ]


def add_postings(
    destination,
    index,
    keys,
):
    """
    Add postings for the supplied keys.
    """

    for key in keys:

        postings = index.get(key)

        if postings:

            destination.update(postings)


def get_name_candidates(
    s1_row,
    name_index,
):

    tokens = safe_tokens(
        s1_row.get(
            "name_tokens",
            "",
        )
    )

    candidates = set()

    add_postings(
        candidates,
        name_index,
        tokens,
    )

    return candidates


def get_address_exact_candidates(
    s1_row,
    address_index,
):

    address = s1_row.get(
        "address_norm",
        "",
    )

    if not address:
        return set()

    postings = address_index.get(
        address
    )

    if not postings:
        return set()

    return set(postings)


def get_address_token_candidates(
    s1_row,
    address_index,
):

    tokens = safe_tokens(
        s1_row.get(
            "address_tokens",
            "",
        )
    )

    candidates = set()

    add_postings(
        candidates,
        address_index,
        tokens,
    )

    return candidates


def get_numeric_candidates(
    s1_row,
    numeric_index,
):

    tokens = safe_tokens(
        s1_row.get(
            "address_numeric",
            "",
        )
    )

    candidates = set()

    add_postings(
        candidates,
        numeric_index,
        tokens,
    )

    return candidates


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidate_ids(
    s1_row,
    name_index,
    address_exact_index,
    address_token_index,
    numeric_index,
):

    name_candidates = get_name_candidates(
        s1_row,
        name_index,
    )

    address_exact_candidates = (
        get_address_exact_candidates(
            s1_row,
            address_exact_index,
        )
    )

    address_token_candidates = (
        get_address_token_candidates(
            s1_row,
            address_token_index,
        )
    )

    numeric_candidates = (
        get_numeric_candidates(
            s1_row,
            numeric_index,
        )
    )

    # Full blocking union.
    candidates = (
        name_candidates
        | address_exact_candidates
        | address_token_candidates
        | numeric_candidates
    )

    return candidates


# ============================================================
# SQLITE LOOKUP
# ============================================================

def fetch_rows(
    conn,
    candidate_ids,
):

    if not candidate_ids:
        return []

    candidate_ids = list(candidate_ids)

    rows = []

    for start in range(
        0,
        len(candidate_ids),
        SQL_BATCH_SIZE,
    ):

        batch = candidate_ids[
            start:start + SQL_BATCH_SIZE
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

            rows.append(
                dict(
                    zip(
                        columns,
                        values,
                    )
                )
            )

    return rows


# ============================================================
# CHEAP PRE-RANKING
# ============================================================

def cheap_score(
    s1,
    candidate,
):

    name_a = s1.get(
        "name_no_suffix",
        "",
    )

    name_b = candidate.get(
        "name_no_suffix",
        "",
    )

    address_a = s1.get(
        "address_norm",
        "",
    )

    address_b = candidate.get(
        "address_norm",
        "",
    )

    name_score = (
        ratio(
            name_a,
            name_b,
        ) / 100.0
        if name_a and name_b
        else 0.0
    )

    address_score = (
        ratio(
            address_a,
            address_b,
        ) / 100.0
        if address_a and address_b
        else 0.0
    )

    country_score = (
        1.0
        if (
            s1.get(
                "country_norm",
                "",
            )
            == candidate.get(
                "country_norm",
                "",
            )
            and s1.get(
                "country_norm",
                "",
            )
        )
        else 0.0
    )

    return (
        0.45 * name_score
        + 0.45 * address_score
        + 0.10 * country_score
    )


def rank_candidates(
    s1_row,
    candidate_rows,
):

    scored = []

    for row in candidate_rows:

        score = cheap_score(
            s1_row,
            row,
        )

        scored.append(
            (
                score,
                row["entity_id"],
            )
        )

    scored.sort(
        reverse=True
    )

    return scored


# ============================================================
# RECALL HELPERS
# ============================================================

def recall_stats(
    candidate_ids,
    truth_ids,
):

    recovered = len(
        set(candidate_ids)
        & set(truth_ids)
    )

    total = len(truth_ids)

    recall = (
        recovered / total
        if total
        else 1.0
    )

    return (
        recovered,
        total,
        recall,
    )


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
        "VALIDATION CANDIDATE PILOT - "
        "BLOCKING / RANKING DIAGNOSTIC"
    )
    print("=" * 70)

    print(
        f"Validation entities : "
        f"{VALIDATION_ENTITY_LIMIT:,}"
    )

    print(
        f"Top K per source    : "
        f"{TOP_K_PER_SOURCE}"
    )

    print(
        f"Address DF cap      : "
        f"{ADDRESS_DF_CAP:,}"
    )

    # --------------------------------------------------------
    # Validation data
    # --------------------------------------------------------

    validation_ids = load_validation_ids()

    truth = load_validation_truth()

    s1_rows = load_validation_s1(
        validation_ids
    )

    # --------------------------------------------------------
    # Load indexes
    # --------------------------------------------------------

    print("\nLoading blocking indexes...")

    name_s2 = load_pickle_index(
        S2_NAME_INDEX
    )

    name_s3 = load_pickle_index(
        S3_NAME_INDEX
    )

    address_exact_s2 = load_pickle_index(
        S2_ADDRESS_EXACT_INDEX
    )

    address_exact_s3 = load_pickle_index(
        S3_ADDRESS_EXACT_INDEX
    )

    address_token_s2 = load_pickle_index(
        S2_ADDRESS_TOKEN_INDEX
    )

    address_token_s3 = load_pickle_index(
        S3_ADDRESS_TOKEN_INDEX
    )

    numeric_s2 = load_pickle_index(
        S2_NUMERIC_INDEX
    )

    numeric_s3 = load_pickle_index(
        S3_NUMERIC_INDEX
    )

    # --------------------------------------------------------
    # Open databases
    # --------------------------------------------------------

    print("\nOpening SQLite databases...")

    conn_s2 = sqlite3.connect(
        f"file:{S2_DB}?mode=ro",
        uri=True,
    )

    conn_s3 = sqlite3.connect(
        f"file:{S3_DB}?mode=ro",
        uri=True,
    )

    # --------------------------------------------------------
    # Output / statistics
    # --------------------------------------------------------

    output_rows = []

    candidate_counts = []

    total_true = 0

    raw_recovered_total = 0

    ranked_recovered_total = 0

    topk_recovered_total = 0

    matched_entities = 0

    raw_complete_entities = 0

    ranked_complete_entities = 0

    topk_complete_entities = 0

    print("\nGenerating candidates...")

    # --------------------------------------------------------
    # Process validation entities
    # --------------------------------------------------------

    for i, s1_id in enumerate(
        validation_ids,
        start=1,
    ):

        s1 = s1_rows.get(
            s1_id
        )

        if s1 is None:
            continue

        actual = truth.get(
            s1_id,
            set(),
        )

        if actual:

            matched_entities += 1

            total_true += len(
                actual
            )

        # ====================================================
        # S2 RAW BLOCKING
        # ====================================================

        s2_ids = generate_candidate_ids(
            s1,
            name_s2,
            address_exact_s2,
            address_token_s2,
            numeric_s2,
        )

        # ====================================================
        # S3 RAW BLOCKING
        # ====================================================

        s3_ids = generate_candidate_ids(
            s1,
            name_s3,
            address_exact_s3,
            address_token_s3,
            numeric_s3,
        )

        # ====================================================
        # RAW UNION
        # ====================================================

        raw_ids = (
            s2_ids
            | s3_ids
        )

        raw_recovered = (
            actual
            & raw_ids
        )

        raw_recovered_total += len(
            raw_recovered
        )

        if (
            actual
            and len(raw_recovered)
            == len(actual)
        ):
            raw_complete_entities += 1

        # ====================================================
        # FETCH ROWS
        # ====================================================

        s2_rows = fetch_rows(
            conn_s2,
            s2_ids,
        )

        s3_rows = fetch_rows(
            conn_s3,
            s3_ids,
        )

        # ====================================================
        # RANKING
        # ====================================================

        ranked_s2_all = rank_candidates(
            s1,
            s2_rows,
        )

        ranked_s3_all = rank_candidates(
            s1,
            s3_rows,
        )

        # ====================================================
        # RECALL AFTER RANKING
        #
        # Before TOP-K.
        # ====================================================

        ranked_ids = {
            candidate_id
            for _, candidate_id
            in ranked_s2_all
        }

        ranked_ids.update(
            candidate_id
            for _, candidate_id
            in ranked_s3_all
        )

        ranked_recovered = (
            actual
            & ranked_ids
        )

        ranked_recovered_total += len(
            ranked_recovered
        )

        if (
            actual
            and len(ranked_recovered)
            == len(actual)
        ):
            ranked_complete_entities += 1

        # ====================================================
        # TOP-K
        # ====================================================

        ranked_s2 = ranked_s2_all[
            :TOP_K_PER_SOURCE
        ]

        ranked_s3 = ranked_s3_all[
            :TOP_K_PER_SOURCE
        ]

        predicted_ids = {
            candidate_id
            for _, candidate_id
            in ranked_s2
        }

        predicted_ids.update(
            candidate_id
            for _, candidate_id
            in ranked_s3
        )

        topk_recovered = (
            actual
            & predicted_ids
        )

        topk_recovered_total += len(
            topk_recovered
        )

        if (
            actual
            and len(topk_recovered)
            == len(actual)
        ):
            topk_complete_entities += 1

        candidate_counts.append(
            len(predicted_ids)
        )

        # ====================================================
        # OUTPUT
        # ====================================================

        for score, candidate_id in ranked_s2:

            output_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "candidate_source": "S2",
                    "block_score": f"{score:.6f}",
                }
            )

        for score, candidate_id in ranked_s3:

            output_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "candidate_source": "S3",
                    "block_score": f"{score:.6f}",
                }
            )

        # ====================================================
        # PROGRESS
        # ====================================================

        if i % 25 == 0:

            elapsed = (
                time.time()
                - start_total
            )

            current_recall = (
                topk_recovered_total
                / total_true
                if total_true
                else 0.0
            )

            raw_recall = (
                raw_recovered_total
                / total_true
                if total_true
                else 0.0
            )

            print(
                f"  processed "
                f"{i:,}/{len(validation_ids):,} | "
                f"rows={len(output_rows):,} | "
                f"raw={raw_recall:.4%} | "
                f"topk={current_recall:.4%} | "
                f"time={elapsed / 60:.1f} min"
            )

    # --------------------------------------------------------
    # Close DBs
    # --------------------------------------------------------

    conn_s2.close()
    conn_s3.close()

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    print("\nWriting candidate file...")

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "block_score",
            ],
            delimiter="\t",
        )

        writer.writeheader()

        writer.writerows(
            output_rows
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    raw_recall = (
        raw_recovered_total
        / total_true
        if total_true
        else 0.0
    )

    ranked_recall = (
        ranked_recovered_total
        / total_true
        if total_true
        else 0.0
    )

    topk_recall = (
        topk_recovered_total
        / total_true
        if total_true
        else 0.0
    )

    raw_completeness = (
        raw_complete_entities
        / matched_entities
        if matched_entities
        else 0.0
    )

    ranked_completeness = (
        ranked_complete_entities
        / matched_entities
        if matched_entities
        else 0.0
    )

    topk_completeness = (
        topk_complete_entities
        / matched_entities
        if matched_entities
        else 0.0
    )

    mean_candidates = (
        statistics.mean(
            candidate_counts
        )
        if candidate_counts
        else 0.0
    )

    median_candidates = (
        statistics.median(
            candidate_counts
        )
        if candidate_counts
        else 0.0
    )

    sorted_counts = sorted(
        candidate_counts
    )

    p95_index = min(
        len(sorted_counts) - 1,
        int(
            len(sorted_counts)
            * 0.95
        ),
    ) if sorted_counts else 0

    p95_candidates = (
        sorted_counts[p95_index]
        if sorted_counts
        else 0
    )

    max_candidates = (
        max(candidate_counts)
        if candidate_counts
        else 0
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BLOCKING / RANKING DIAGNOSTIC")
    print("=" * 70)

    print(
        f"Validation entities       : "
        f"{len(validation_ids):,}"
    )

    print(
        f"Matched validation S1     : "
        f"{matched_entities:,}"
    )

    print(
        f"True links                : "
        f"{total_true:,}"
    )

    print()

    print("1. RAW BLOCKER UNION")
    print(
        f"   Recovered links        : "
        f"{raw_recovered_total:,}"
    )
    print(
        f"   Candidate recall       : "
        f"{raw_recall:.6%}"
    )
    print(
        f"   Complete entity recall : "
        f"{raw_completeness:.6%}"
    )

    print()

    print("2. AFTER CHEAP RANKING")
    print(
        f"   Recovered links        : "
        f"{ranked_recovered_total:,}"
    )
    print(
        f"   Candidate recall       : "
        f"{ranked_recall:.6%}"
    )
    print(
        f"   Complete entity recall : "
        f"{ranked_completeness:.6%}"
    )

    print()

    print(
        f"3. AFTER TOP-{TOP_K_PER_SOURCE}"
    )
    print(
        f"   Recovered links        : "
        f"{topk_recovered_total:,}"
    )
    print(
        f"   Candidate recall       : "
        f"{topk_recall:.6%}"
    )
    print(
        f"   Complete entity recall : "
        f"{topk_completeness:.6%}"
    )

    print()

    print("RECALL LOST")
    print(
        f"   Raw -> Ranking         : "
        f"{raw_recall - ranked_recall:.6%}"
    )
    print(
        f"   Ranking -> Top-K       : "
        f"{ranked_recall - topk_recall:.6%}"
    )

    print()

    print("CANDIDATE VOLUME")
    print(
        f"   Mean candidates/entity : "
        f"{mean_candidates:,.2f}"
    )
    print(
        f"   Median                 : "
        f"{median_candidates:,.2f}"
    )
    print(
        f"   P95                    : "
        f"{p95_candidates:,.2f}"
    )
    print(
        f"   Max                    : "
        f"{max_candidates:,}"
    )

    print()

    print(
        f"Candidate rows written    : "
        f"{len(output_rows):,}"
    )

    print(
        f"Output                    : "
        f"{OUTPUT_FILE}"
    )

    print()

    print(
        f"Total runtime             : "
        f"{(time.time() - start_total) / 60:.2f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()