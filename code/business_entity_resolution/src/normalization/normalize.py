from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from unidecode import unidecode


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

CHUNK_SIZE = 100_000

NAME_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "ltd",
    "limited",
    "llp",
    "plc",
    "pvt",
    "private",
    "privatelimited",
    "private limited",
}


# ---------------------------------------------------------------------
# Basic text utilities
# ---------------------------------------------------------------------

def clean_text(value: object) -> str:
    """
    Convert a value to a safe string representation.

    Original data is never modified by this function.
    """
    if value is None:
        return ""

    text = str(value)

    if text.lower() in {"nan", "none", "<na>"}:
        return ""

    return text.strip()


def unicode_normalize(text: str) -> str:
    """
    Normalize Unicode characters while preserving the general meaning.
    """
    text = clean_text(text)

    # Compatibility normalization.
    text = unicodedata.normalize("NFKC", text)

    # Case-insensitive matching.
    text = text.casefold()

    return text


def accent_normalize(text: str) -> str:
    """
    Remove accents/diacritics.

    Example:
        café -> cafe
        naïve -> naive
    """
    text = unicode_normalize(text)

    text = unicodedata.normalize("NFKD", text)

    return "".join(
        char for char in text
        if not unicodedata.combining(char)
    )


def transliterate(text: str) -> str:
    """
    Convert non-Latin scripts to a Latin representation where possible.
    """
    text = unicode_normalize(text)

    return unidecode(text).casefold().strip()


def normalize_spaces(text: str) -> str:
    """
    Collapse repeated whitespace.
    """
    return re.sub(r"\s+", " ", text).strip()


def alphanumeric_only(text: str) -> str:
    """
    Keep only letters and digits.
    """
    return re.sub(r"[^a-z0-9]+", "", text)


def tokenize(text: str) -> list[str]:
    """
    Produce normalized alphanumeric tokens.
    """
    text = transliterate(text)

    tokens = re.findall(r"[a-z0-9]+", text)

    return tokens


# ---------------------------------------------------------------------
# Name views
# ---------------------------------------------------------------------

def normalize_name_views(value: object) -> dict[str, str]:
    """
    Generate several representations of a business name.
    """

    raw = clean_text(value)

    basic = accent_normalize(raw)
    basic = normalize_spaces(basic)

    translit = transliterate(raw)
    translit = normalize_spaces(translit)

    tokens = tokenize(raw)

    # Remove common legal suffixes only for the suffix-stripped view.
    stripped_tokens = [
        token
        for token in tokens
        if token not in NAME_SUFFIXES
    ]

    return {
        "name_norm": basic,
        "name_compact": alphanumeric_only(translit),
        "name_tokens": " ".join(tokens),
        "name_sorted": " ".join(sorted(tokens)),
        "name_no_suffix": " ".join(stripped_tokens),
        "name_translit": translit,
    }


# ---------------------------------------------------------------------
# Address views
# ---------------------------------------------------------------------

def normalize_address_views(value: object) -> dict[str, str]:
    """
    Generate several representations of a business address.
    """

    raw = clean_text(value)

    basic = accent_normalize(raw)
    basic = normalize_spaces(basic)

    translit = transliterate(raw)
    translit = normalize_spaces(translit)

    tokens = tokenize(raw)

    return {
        "address_norm": basic,
        "address_compact": alphanumeric_only(translit),
        "address_tokens": " ".join(tokens),
        "address_sorted": " ".join(sorted(tokens)),
        "address_translit": translit,
    }


# ---------------------------------------------------------------------
# Country
# ---------------------------------------------------------------------

def normalize_country(value: object) -> str:
    """
    Country is treated as an open-set value.

    We normalize formatting but do not restrict it to US/India/etc.
    """

    text = clean_text(value)

    text = unicode_normalize(text)

    return normalize_spaces(text)


# ---------------------------------------------------------------------
# DataFrame transformation
# ---------------------------------------------------------------------

def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized representations to one DataFrame chunk.
    """

    name_views = df["business_name"].map(normalize_name_views)
    name_views = pd.DataFrame(name_views.tolist(), index=df.index)

    address_views = df["business_address"].map(normalize_address_views)
    address_views = pd.DataFrame(address_views.tolist(), index=df.index)

    country_norm = df["country"].map(normalize_country)

    normalized = pd.concat(
        [
            df,
            name_views,
            address_views,
            country_norm.rename("country_norm"),
        ],
        axis=1,
    )

    return normalized


# ---------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------

def normalize_file(
    input_path: Path,
    output_path: Path,
    sample_rows: int | None = None,
) -> None:
    """
    Normalize one TSV file using chunks so that large datasets
    do not need to fit completely into memory.
    """

    print()
    print("=" * 70)
    print(f"INPUT : {input_path}")
    print(f"OUTPUT: {output_path}")
    print("=" * 70)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    # Remove an old output so that repeated runs do not append
    # duplicate headers/data.
    if output_path.exists():
        output_path.unlink()

    total_rows = 0
    first_chunk = True

    for chunk in pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
    ):

        if sample_rows is not None:
            remaining = sample_rows - total_rows

            if remaining <= 0:
                break

            chunk = chunk.iloc[:remaining]

        normalized = normalize_chunk(chunk)

        normalized.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )

        first_chunk = False
        total_rows += len(normalized)

        print(f"Processed rows: {total_rows:,}")

        if sample_rows is not None and total_rows >= sample_rows:
            break

    print(f"Finished: {total_rows:,} rows")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Multi-view business entity normalization"
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only the first N rows of each file.",
    )

    parser.add_argument(
        "--source",
        choices=["train", "test", "all"],
        default="train",
        help="Which dataset split to normalize.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]

    dataset_root = project_root / "dataset"
    output_root = project_root / "data" / "normalized"

    source1 = ["source1"]
    source2 = ["source2"]
    source3 = ["source3"]

    splits = []

    if args.source in {"train", "all"}:
        splits.append("train")

    if args.source in {"test", "all"}:
        splits.append("test")

    for split in splits:

        for source in ["source1", "source2", "source3"]:

            input_path = (
                dataset_root
                / split
                / f"{split}_{source}.tsv"
            )

            output_path = (
                output_root
                / split
                / f"{split}_{source}_normalized.tsv"
            )

            normalize_file(
                input_path=input_path,
                output_path=output_path,
                sample_rows=args.sample,
            )

    print()
    print("=" * 70)
    print("PHASE 3 NORMALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()