from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "training_pairs_sample.tsv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "training_features_sample.tsv"
)


# ============================================================
# HELPERS
# ============================================================

def safe_text(value):
    if value is None:
        return ""

    return str(value)


def tokens(value):
    value = safe_text(value).strip()

    if not value:
        return set()

    return set(value.split())


def numeric_tokens(value):
    value = safe_text(value)

    # Fast manual numeric extraction.
    result = set()
    current = []

    for char in value:

        if char.isdigit():
            current.append(char)

        elif current:
            result.add("".join(current))
            current = []

    if current:
        result.add("".join(current))

    return result


def jaccard(a, b):

    a = tokens(a)
    b = tokens(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def overlap(a, b):

    a = tokens(a)
    b = tokens(b)

    if not a or not b:
        return 0.0

    return len(a & b) / min(
        len(a),
        len(b),
    )


def numeric_jaccard(a, b):

    a = numeric_tokens(a)
    b = numeric_tokens(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def numeric_overlap(a, b):

    a = numeric_tokens(a)
    b = numeric_tokens(b)

    if not a or not b:
        return 0.0

    return len(a & b) / min(
        len(a),
        len(b),
    )


def exact(a, b):

    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0

    return int(a == b)


def length_ratio(a, b):

    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    return min(
        len(a),
        len(b),
    ) / max(
        len(a),
        len(b),
    )


def length_diff(a, b):

    return abs(
        len(safe_text(a))
        - len(safe_text(b))
    )


def token_count_diff(a, b):

    return abs(
        len(tokens(a))
        - len(tokens(b))
    )


def edit_similarity(a, b):

    a = safe_text(a)
    b = safe_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


# ============================================================
# FEATURE BUILDER
# ============================================================

def build_features(row):

    # --------------------------------------------------------
    # S1
    # --------------------------------------------------------

    name = safe_text(row["s1_name_norm"])
    compact = safe_text(row["s1_name_compact"])
    no_suffix = safe_text(row["s1_name_no_suffix"])
    sorted_name = safe_text(row["s1_name_sorted"])
    translit_name = safe_text(row["s1_name_translit"])

    address = safe_text(row["s1_address_norm"])
    address_compact = safe_text(
        row["s1_address_compact"]
    )
    address_sorted = safe_text(
        row["s1_address_sorted"]
    )
    address_translit = safe_text(
        row["s1_address_translit"]
    )

    country = safe_text(row["s1_country_norm"])

    # --------------------------------------------------------
    # Candidate
    # --------------------------------------------------------

    c_name = safe_text(
        row["candidate_name_norm"]
    )
    c_compact = safe_text(
        row["candidate_name_compact"]
    )
    c_no_suffix = safe_text(
        row["candidate_name_no_suffix"]
    )
    c_sorted_name = safe_text(
        row["candidate_name_sorted"]
    )
    c_translit_name = safe_text(
        row["candidate_name_translit"]
    )

    c_address = safe_text(
        row["candidate_address_norm"]
    )
    c_address_compact = safe_text(
        row["candidate_address_compact"]
    )
    c_address_sorted = safe_text(
        row["candidate_address_sorted"]
    )
    c_address_translit = safe_text(
        row["candidate_address_translit"]
    )

    c_country = safe_text(
        row["candidate_country_norm"]
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    return {

        # ====================================================
        # BLOCKER PROVENANCE
        # ====================================================

        "block_name_token":
            int(row["block_name_token"]),

        "block_exact_address":
            int(row["block_exact_address"]),

        # ====================================================
        # NAME EXACT
        # ====================================================

        "name_norm_exact":
            exact(name, c_name),

        "name_compact_exact":
            exact(compact, c_compact),

        "name_no_suffix_exact":
            exact(no_suffix, c_no_suffix),

        "name_sorted_exact":
            exact(sorted_name, c_sorted_name),

        "name_translit_exact":
            exact(translit_name, c_translit_name),

        # ====================================================
        # NAME TOKEN
        # ====================================================

        "name_jaccard":
            jaccard(name, c_name),

        "name_overlap":
            overlap(name, c_name),

        "name_token_intersection":
            len(
                tokens(name)
                & tokens(c_name)
            ),

        # ====================================================
        # NAME EDIT
        # ====================================================

        "name_edit_similarity":
            edit_similarity(name, c_name),

        "name_compact_edit_similarity":
            edit_similarity(
                compact,
                c_compact,
            ),

        "name_no_suffix_edit_similarity":
            edit_similarity(
                no_suffix,
                c_no_suffix,
            ),

        "name_translit_edit_similarity":
            edit_similarity(
                translit_name,
                c_translit_name,
            ),

        "name_sorted_edit_similarity":
            edit_similarity(
                sorted_name,
                c_sorted_name,
            ),

        # ====================================================
        # NAME STRUCTURE
        # ====================================================

        "name_length_ratio":
            length_ratio(
                name,
                c_name,
            ),

        "name_length_diff":
            length_diff(
                name,
                c_name,
            ),

        "name_token_count_diff":
            token_count_diff(
                name,
                c_name,
            ),

        # ====================================================
        # ADDRESS EXACT
        # ====================================================

        "address_norm_exact":
            exact(
                address,
                c_address,
            ),

        "address_compact_exact":
            exact(
                address_compact,
                c_address_compact,
            ),

        "address_sorted_exact":
            exact(
                address_sorted,
                c_address_sorted,
            ),

        "address_translit_exact":
            exact(
                address_translit,
                c_address_translit,
            ),

        # ====================================================
        # ADDRESS TOKEN
        # ====================================================

        "address_jaccard":
            jaccard(
                address,
                c_address,
            ),

        "address_overlap":
            overlap(
                address,
                c_address,
            ),

        "address_token_intersection":
            len(
                tokens(address)
                & tokens(c_address)
            ),

        # ====================================================
        # ADDRESS EDIT
        # ====================================================

        "address_edit_similarity":
            edit_similarity(
                address,
                c_address,
            ),

        "address_compact_edit_similarity":
            edit_similarity(
                address_compact,
                c_address_compact,
            ),

        # ====================================================
        # ADDRESS NUMERIC
        # ====================================================

        "address_numeric_jaccard":
            numeric_jaccard(
                address,
                c_address,
            ),

        "address_numeric_overlap":
            numeric_overlap(
                address,
                c_address,
            ),

        "address_numeric_exact":
            int(
                numeric_tokens(address)
                == numeric_tokens(c_address)
                and bool(
                    numeric_tokens(address)
                )
            ),

        # ====================================================
        # ADDRESS STRUCTURE
        # ====================================================

        "address_length_ratio":
            length_ratio(
                address,
                c_address,
            ),

        "address_length_diff":
            length_diff(
                address,
                c_address,
            ),

        "address_token_count_diff":
            token_count_diff(
                address,
                c_address,
            ),

        # ====================================================
        # COUNTRY
        # ====================================================

        "country_exact":
            exact(
                country,
                c_country,
            ),

        "country_missing_either":
            int(
                not country
                or not c_country
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 5C — PAIR FEATURE GENERATION")
    print("=" * 70)

    print(f"Input:  {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    # --------------------------------------------------------
    # Load training pairs in chunks
    # --------------------------------------------------------

    first_write = True

    total_rows = 0

    feature_names = None

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            INPUT_PATH,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=25_000,
        ),
        start=1,
    ):

        print(
            f"\nProcessing chunk {chunk_number} "
            f"({len(chunk):,} rows)..."
        )

        feature_rows = []

        for _, row in chunk.iterrows():

            features = build_features(row)

            # Keep identifiers/label separately.
            features[
                "source1_entity_id"
            ] = row["source1_entity_id"]

            features[
                "candidate_entity_id"
            ] = row["candidate_entity_id"]

            features[
                "candidate_source"
            ] = row["candidate_source"]

            features[
                "label"
            ] = int(row["label"])

            feature_rows.append(features)

        feature_df = pd.DataFrame(
            feature_rows
        )

        # Ensure stable column order.
        if feature_names is None:

            metadata_columns = [
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "label",
            ]

            feature_names = [
                c
                for c in feature_df.columns
                if c not in metadata_columns
            ]

            column_order = (
                metadata_columns
                + feature_names
            )

        feature_df = feature_df[
            column_order
        ]

        feature_df.to_csv(
            OUTPUT_PATH,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
        )

        first_write = False

        total_rows += len(feature_df)

        print(
            f"Written: {total_rows:,}"
        )

        del feature_rows
        del feature_df
        del chunk

        gc.collect()

    print("\n" + "=" * 70)
    print("PHASE 5C COMPLETE")
    print("=" * 70)

    print(
        f"Feature rows written: {total_rows:,}"
    )

    print(
        f"Feature count: {len(feature_names):,}"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()