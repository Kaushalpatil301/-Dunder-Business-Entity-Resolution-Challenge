from __future__ import annotations

import math
import re
from typing import Iterable, Set


# ============================================================
# BASIC TOKEN HELPERS
# ============================================================

def safe_text(value) -> str:
    if value is None:
        return ""
    return str(value)


def token_set(value: str) -> Set[str]:
    value = safe_text(value).strip()

    if not value:
        return set()

    return set(value.split())


def numeric_tokens(value: str) -> Set[str]:
    value = safe_text(value)

    return set(
        re.findall(r"\d+", value)
    )


# ============================================================
# EXACT FEATURES
# ============================================================

def exact_match(a: str, b: str) -> int:
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0

    return int(a == b)


# ============================================================
# TOKEN FEATURES
# ============================================================

def jaccard_similarity(a: str, b: str) -> float:
    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)

    return intersection / union if union else 0.0


def overlap_coefficient(a: str, b: str) -> float:
    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)

    return intersection / min(
        len(a_tokens),
        len(b_tokens),
    )


def token_intersection_count(a: str, b: str) -> int:
    return len(
        token_set(a) & token_set(b)
    )


# ============================================================
# NUMERIC ADDRESS FEATURES
# ============================================================

def numeric_jaccard(a: str, b: str) -> float:
    a_nums = numeric_tokens(a)
    b_nums = numeric_tokens(b)

    if not a_nums and not b_nums:
        return 1.0

    if not a_nums or not b_nums:
        return 0.0

    return len(a_nums & b_nums) / len(
        a_nums | b_nums
    )


def numeric_overlap(a: str, b: str) -> float:
    a_nums = numeric_tokens(a)
    b_nums = numeric_tokens(b)

    if not a_nums or not b_nums:
        return 0.0

    return len(a_nums & b_nums) / min(
        len(a_nums),
        len(b_nums),
    )


def numeric_exact_match(a: str, b: str) -> int:
    a_nums = numeric_tokens(a)
    b_nums = numeric_tokens(b)

    if not a_nums or not b_nums:
        return 0

    return int(a_nums == b_nums)


# ============================================================
# CHARACTER SIMILARITY
# ============================================================

def levenshtein_distance(a: str, b: str) -> int:
    """
    Standard dynamic-programming Levenshtein distance.
    """

    a = safe_text(a)
    b = safe_text(b)

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    # Keep b as the shorter string to reduce memory.
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):

        current = [i]

        for j, cb in enumerate(b, start=1):

            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    replace_cost,
                )
            )

        previous = current

    return previous[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    a = safe_text(a)
    b = safe_text(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    distance = levenshtein_distance(a, b)

    return 1.0 - (
        distance / max(len(a), len(b))
    )


# ============================================================
# LENGTH / STRUCTURAL FEATURES
# ============================================================

def length_ratio(a: str, b: str) -> float:
    a = safe_text(a)
    b = safe_text(b)

    if not a or not b:
        return 0.0

    return min(len(a), len(b)) / max(
        len(a),
        len(b),
    )


def absolute_length_difference(a: str, b: str) -> int:
    return abs(
        len(safe_text(a))
        - len(safe_text(b))
    )


def token_count_difference(a: str, b: str) -> int:
    return abs(
        len(token_set(a))
        - len(token_set(b))
    )


# ============================================================
# PAIR FEATURE GENERATOR
# ============================================================

def build_pair_features(
    s1: dict,
    candidate: dict,
) -> dict:
    """
    Build pairwise features for one S1 <-> S2/S3 candidate.

    Expected fields:

    entity_id
    name_norm
    name_compact
    name_no_suffix
    name_sorted
    name_translit
    address_norm
    address_compact
    address_sorted
    address_translit
    country_norm
    """

    s1_name = safe_text(s1.get("name_norm"))
    c_name = safe_text(candidate.get("name_norm"))

    s1_compact = safe_text(s1.get("name_compact"))
    c_compact = safe_text(candidate.get("name_compact"))

    s1_no_suffix = safe_text(
        s1.get("name_no_suffix")
    )
    c_no_suffix = safe_text(
        candidate.get("name_no_suffix")
    )

    s1_sorted = safe_text(
        s1.get("name_sorted")
    )
    c_sorted = safe_text(
        candidate.get("name_sorted")
    )

    s1_translit = safe_text(
        s1.get("name_translit")
    )
    c_translit = safe_text(
        candidate.get("name_translit")
    )

    s1_address = safe_text(
        s1.get("address_norm")
    )
    c_address = safe_text(
        candidate.get("address_norm")
    )

    s1_address_compact = safe_text(
        s1.get("address_compact")
    )
    c_address_compact = safe_text(
        candidate.get("address_compact")
    )

    s1_address_sorted = safe_text(
        s1.get("address_sorted")
    )
    c_address_sorted = safe_text(
        candidate.get("address_sorted")
    )

    s1_address_translit = safe_text(
        s1.get("address_translit")
    )
    c_address_translit = safe_text(
        candidate.get("address_translit")
    )

    s1_country = safe_text(
        s1.get("country_norm")
    )
    c_country = safe_text(
        candidate.get("country_norm")
    )

    features = {
        # ----------------------------------------------------
        # Identity
        # ----------------------------------------------------

        "candidate_source": candidate.get(
            "source",
            "",
        ),

        # ----------------------------------------------------
        # NAME — EXACT
        # ----------------------------------------------------

        "name_norm_exact":
            exact_match(
                s1_name,
                c_name,
            ),

        "name_compact_exact":
            exact_match(
                s1_compact,
                c_compact,
            ),

        "name_no_suffix_exact":
            exact_match(
                s1_no_suffix,
                c_no_suffix,
            ),

        "name_sorted_exact":
            exact_match(
                s1_sorted,
                c_sorted,
            ),

        "name_translit_exact":
            exact_match(
                s1_translit,
                c_translit,
            ),

        # ----------------------------------------------------
        # NAME — TOKEN
        # ----------------------------------------------------

        "name_jaccard":
            jaccard_similarity(
                s1_name,
                c_name,
            ),

        "name_overlap":
            overlap_coefficient(
                s1_name,
                c_name,
            ),

        "name_token_intersection":
            token_intersection_count(
                s1_name,
                c_name,
            ),

        # ----------------------------------------------------
        # NAME — CHARACTER
        # ----------------------------------------------------

        "name_edit_similarity":
            levenshtein_similarity(
                s1_name,
                c_name,
            ),

        "name_compact_edit_similarity":
            levenshtein_similarity(
                s1_compact,
                c_compact,
            ),

        "name_no_suffix_edit_similarity":
            levenshtein_similarity(
                s1_no_suffix,
                c_no_suffix,
            ),

        "name_translit_edit_similarity":
            levenshtein_similarity(
                s1_translit,
                c_translit,
            ),

        # ----------------------------------------------------
        # NAME — STRUCTURE
        # ----------------------------------------------------

        "name_length_ratio":
            length_ratio(
                s1_name,
                c_name,
            ),

        "name_length_diff":
            absolute_length_difference(
                s1_name,
                c_name,
            ),

        "name_token_count_diff":
            token_count_difference(
                s1_name,
                c_name,
            ),

        "name_sorted_edit_similarity":
            levenshtein_similarity(
                s1_sorted,
                c_sorted,
            ),

        # ----------------------------------------------------
        # ADDRESS — EXACT
        # ----------------------------------------------------

        "address_norm_exact":
            exact_match(
                s1_address,
                c_address,
            ),

        "address_compact_exact":
            exact_match(
                s1_address_compact,
                c_address_compact,
            ),

        "address_sorted_exact":
            exact_match(
                s1_address_sorted,
                c_address_sorted,
            ),

        "address_translit_exact":
            exact_match(
                s1_address_translit,
                c_address_translit,
            ),

        # ----------------------------------------------------
        # ADDRESS — TOKEN
        # ----------------------------------------------------

        "address_jaccard":
            jaccard_similarity(
                s1_address,
                c_address,
            ),

        "address_overlap":
            overlap_coefficient(
                s1_address,
                c_address,
            ),

        "address_token_intersection":
            token_intersection_count(
                s1_address,
                c_address,
            ),

        # ----------------------------------------------------
        # ADDRESS — CHARACTER
        # ----------------------------------------------------

        "address_edit_similarity":
            levenshtein_similarity(
                s1_address,
                c_address,
            ),

        "address_compact_edit_similarity":
            levenshtein_similarity(
                s1_address_compact,
                c_address_compact,
            ),

        # ----------------------------------------------------
        # ADDRESS — NUMERIC
        # ----------------------------------------------------

        "address_numeric_jaccard":
            numeric_jaccard(
                s1_address,
                c_address,
            ),

        "address_numeric_overlap":
            numeric_overlap(
                s1_address,
                c_address,
            ),

        "address_numeric_exact":
            numeric_exact_match(
                s1_address,
                c_address,
            ),

        # ----------------------------------------------------
        # ADDRESS — STRUCTURE
        # ----------------------------------------------------

        "address_length_ratio":
            length_ratio(
                s1_address,
                c_address,
            ),

        "address_length_diff":
            absolute_length_difference(
                s1_address,
                c_address,
            ),

        "address_token_count_diff":
            token_count_difference(
                s1_address,
                c_address,
            ),

        # ----------------------------------------------------
        # COUNTRY
        # ----------------------------------------------------

        "country_exact":
            exact_match(
                s1_country,
                c_country,
            ),

        "country_missing_either":
            int(
                not s1_country
                or not c_country
            ),
    }

    return features