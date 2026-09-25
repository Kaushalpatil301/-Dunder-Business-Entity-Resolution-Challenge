"""Phase 1 exploratory analysis for the business entity resolution data."""

from __future__ import annotations

import csv
import gc
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import (
    GROUND_TRUTH,
    PROJECT_ROOT,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)


SOURCE_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")
TEXT_COLUMNS = ("business_name", "business_address", "country")
NA_TOKENS = ("NA", "N/A", "NULL", "null", "NaN", "nan", "None")
LEGAL_SUFFIX_PATTERN = (
    r"(?i)(?:\b(?:co|company|corp|corporation|inc|incorporated|llc|llp|ltd|limited|"
    r"plc|pvt|gmbh|sa|sas|sarl)\b|प्राइवेट\s+लिमिटेड|लिमिटेड)"
)
PUNCTUATION_PATTERN = r"[^\w\s]"
DIGIT_PATTERN = r"\d"


class TextReport:
    """Small helper for building a readable plain-text report."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def heading(self, title: str) -> None:
        if self.lines:
            self.lines.append("")
        self.lines.extend((title, "=" * len(title)))

    def subheading(self, title: str) -> None:
        self.lines.extend(("", title, "-" * len(title)))

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def _percentage(count: int, total: int) -> float:
    return 100.0 * count / total if total else 0.0


def _read_source(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=list(NA_TOKENS),
    )


def _validate_columns(
    actual_columns: pd.Index, expected_columns: tuple[str, ...], label: str
) -> None:
    missing = [column for column in expected_columns if column not in actual_columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _missing_statistics(df: pd.DataFrame, report: TextReport) -> dict[str, int]:
    results: dict[str, int] = {}
    report.add("Missing values and blank strings (measured separately):")
    for column in TEXT_COLUMNS:
        missing = int(df[column].isna().sum())
        blank = int((df[column].notna() & df[column].str.strip().eq("")).sum())
        results[f"missing_{column}"] = missing
        results[f"blank_{column}"] = blank
        report.add(
            f"  {column}: missing={missing:,} ({_percentage(missing, len(df)):.4f}%), "
            f"blank={blank:,} ({_percentage(blank, len(df)):.4f}%)"
        )
    return results


def _country_statistics(df: pd.DataFrame, report: TextReport) -> dict[str, object]:
    country = df["country"]
    displayed = country.fillna("<MISSING>").astype(str)
    displayed = displayed.mask(displayed.str.strip().eq(""), "<BLANK>")
    counts = displayed.value_counts(dropna=False)

    unknown_values = {"unknown", "unk", "n/a", "na", "none", "null", "<missing>", "<blank>"}
    missing_unknown = int(displayed.str.strip().str.lower().isin(unknown_values).sum())

    report.add("Country distribution:")
    for country_value, count in counts.items():
        report.add(
            f"  {country_value}: {count:,} ({_percentage(int(count), len(df)):.4f}%)"
        )
    report.add(
        f"  Missing/blank/unknown combined: {missing_unknown:,} "
        f"({_percentage(missing_unknown, len(df)):.4f}%)"
    )
    return {"counts": counts.to_dict(), "missing_unknown": missing_unknown}


def _duplicate_statistics(df: pd.DataFrame, report: TextReport) -> None:
    entity = df["entity_id"]
    name = df["business_name"]
    address = df["business_address"]

    valid_entity = entity.notna() & entity.str.strip().ne("")
    valid_name = name.notna() & name.str.strip().ne("")
    valid_address = address.notna() & address.str.strip().ne("")
    valid_pair = valid_name & valid_address

    report.add("Unique values and exact duplicates:")
    report.add(f"  Unique nonblank entity_id values: {entity[valid_entity].nunique():,}")
    report.add(
        f"  Duplicate nonblank entity_id rows after first: "
        f"{int(entity[valid_entity].duplicated().sum()):,}"
    )
    report.add(
        f"  Exact duplicate nonblank business_name rows after first: "
        f"{int(name[valid_name].duplicated().sum()):,}"
    )
    report.add(
        f"  Exact duplicate nonblank business_address rows after first: "
        f"{int(address[valid_address].duplicated().sum()):,}"
    )
    exact_pairs = df.loc[valid_pair, ["business_name", "business_address"]]
    report.add(
        f"  Exact duplicate nonblank name+address rows after first: "
        f"{int(exact_pairs.duplicated().sum()):,}"
    )
    del exact_pairs

    normalized_name = name.str.strip().str.lower()
    normalized_address = address.str.strip().str.lower()
    normalized_pairs = pd.DataFrame(
        {"business_name": normalized_name[valid_pair], "business_address": normalized_address[valid_pair]}
    )
    report.add("Diagnostic duplicates after whitespace trimming + lowercase only:")
    report.add(
        f"  Duplicate nonblank business_name rows after first: "
        f"{int(normalized_name[valid_name].duplicated().sum()):,}"
    )
    report.add(
        f"  Duplicate nonblank business_address rows after first: "
        f"{int(normalized_address[valid_address].duplicated().sum()):,}"
    )
    report.add(
        f"  Duplicate nonblank name+address rows after first: "
        f"{int(normalized_pairs.duplicated().sum()):,}"
    )


def _describe_numeric(values: pd.Series) -> str:
    if values.empty:
        return "no nonblank values"
    return (
        f"mean={values.mean():.2f}, median={values.median():.2f}, "
        f"min={int(values.min())}, max={int(values.max())}"
    )


def _noise_statistics(df: pd.DataFrame, report: TextReport) -> None:
    names = df["business_name"].dropna().astype(str).str.strip()
    names = names[names.ne("")]
    name_lengths = names.str.len()

    report.add("Business-name noise:")
    report.add(f"  Lengths: {_describe_numeric(name_lengths)}")
    report.add(f"  Contains punctuation: {int(names.str.contains(PUNCTUATION_PATTERN, regex=True).sum()):,}")
    report.add(
        f"  Contains digits: {int(names.str.contains(DIGIT_PATTERN, regex=True).sum()):,}"
    )
    report.add(
        f"  Contains a measured legal/company suffix pattern: "
        f"{int(names.str.contains(LEGAL_SUFFIX_PATTERN, regex=True).sum()):,}"
    )
    report.add(
        "  Patterns measured: co, company, corp/corporation, inc/incorporated, LLC, "
        "LLP, ltd/limited, PLC, Pvt, GmbH, SA, SAS, SARL, and common Hindi limited forms."
    )

    addresses_all = df["business_address"].dropna().astype(str).str.strip()
    blank_addresses = int(addresses_all.eq("").sum())
    addresses = addresses_all[addresses_all.ne("")]
    address_lengths = addresses.str.len()
    token_counts = addresses.str.split().str.len()

    report.add("Address noise:")
    report.add(f"  Lengths: {_describe_numeric(address_lengths)}")
    report.add(
        f"  Contains digits: {int(addresses.str.contains(DIGIT_PATTERN, regex=True).sum()):,}"
    )
    report.add(
        f"  Contains punctuation: "
        f"{int(addresses.str.contains(PUNCTUATION_PATTERN, regex=True).sum()):,}"
    )
    report.add(f"  Blank addresses: {blank_addresses:,}")
    report.add(f"  Very short nonblank addresses (<10 characters): {int(address_lengths.lt(10).sum()):,}")
    report.add(f"  Whitespace token counts: {_describe_numeric(token_counts)}")


def analyze_source(
    label: str,
    path: Path,
    report: TextReport,
    sample_ids: set[str],
    full_diagnostics: bool,
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    print(f"Analyzing {label}: {path}", flush=True)
    df = _read_source(path)
    _validate_columns(df.columns, SOURCE_COLUMNS, label)

    report.subheading(label)
    report.add(f"Path: {path}")
    report.add(f"Rows: {len(df):,}")
    report.add(f"Columns: {len(df.columns):,}")
    report.add(f"Column names: {', '.join(df.columns)}")
    report.add("Data types:")
    for column, dtype in df.dtypes.items():
        report.add(f"  {column}: {dtype}")
    memory_bytes = int(df.memory_usage(index=True, deep=True).sum())
    report.add(f"Deep memory usage: {memory_bytes / (1024 ** 2):,.2f} MiB")
    report.add("Required columns present: yes")

    missing = _missing_statistics(df, report)
    country = _country_statistics(df, report)
    if full_diagnostics:
        _duplicate_statistics(df, report)
        _noise_statistics(df, report)

    sample_rows = df[df["entity_id"].isin(sample_ids)].drop_duplicates("entity_id")
    details = {
        str(row.entity_id): {
            "name": "<MISSING>" if pd.isna(row.business_name) else str(row.business_name),
            "address": "<MISSING>" if pd.isna(row.business_address) else str(row.business_address),
        }
        for row in sample_rows.itertuples(index=False)
    }
    summary: dict[str, object] = {
        "rows": len(df),
        "missing_name": missing["missing_business_name"] + missing["blank_business_name"],
        "missing_address": missing["missing_business_address"] + missing["blank_business_address"],
        "countries": country["counts"],
    }
    del sample_rows, df
    gc.collect()
    return summary, details


def _median_from_counts(cardinality_counts: Counter[int], total: int) -> float:
    if not total:
        return 0.0
    middle_positions = ((total - 1) // 2, total // 2)
    values: list[int] = []
    cumulative = 0
    for cardinality in sorted(cardinality_counts):
        cumulative += cardinality_counts[cardinality]
        for position in middle_positions[len(values) :]:
            if cumulative > position:
                values.append(cardinality)
    return sum(values) / 2


def analyze_ground_truth(report: TextReport) -> dict[str, object]:
    print(f"Analyzing ground truth: {GROUND_TRUTH}", flush=True)
    cardinalities: Counter[int] = Counter()
    total_entities = 0
    total_links = 0
    maximum = 0
    unknown_prefix_links = 0
    positive_samples: list[tuple[str, str]] = []
    singleton_samples: list[str] = []

    with tempfile.TemporaryDirectory(prefix="phase1_eda_") as temp_dir:
        database_path = Path(temp_dir) / "side_counts.sqlite3"
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE side_counts (side TEXT, entity_id TEXT, s1_count INTEGER, "
            "PRIMARY KEY (side, entity_id)) WITHOUT ROWID"
        )
        upsert = (
            "INSERT INTO side_counts(side, entity_id, s1_count) VALUES (?, ?, 1) "
            "ON CONFLICT(side, entity_id) DO UPDATE SET s1_count = s1_count + 1"
        )
        batch: list[tuple[str, str]] = []

        with GROUND_TRUTH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            _validate_columns(pd.Index(reader.fieldnames or []), GROUND_TRUTH_COLUMNS, "ground truth")
            for row in reader:
                source1_id = row["source1_entity_id"].strip()
                matched_ids = {
                    entity_id.strip()
                    for entity_id in row["matched_entity_ids"].split(",")
                    if entity_id.strip()
                }
                cardinality = len(matched_ids)
                cardinalities[cardinality] += 1
                total_entities += 1
                total_links += cardinality
                maximum = max(maximum, cardinality)

                if cardinality == 0 and len(singleton_samples) < 10:
                    singleton_samples.append(source1_id)

                for matched_id in matched_ids:
                    if len(positive_samples) < 20:
                        positive_samples.append((source1_id, matched_id))
                    if matched_id.startswith("S2-"):
                        batch.append(("S2", matched_id))
                    elif matched_id.startswith("S3-"):
                        batch.append(("S3", matched_id))
                    else:
                        unknown_prefix_links += 1

                if len(batch) >= 50_000:
                    connection.executemany(upsert, batch)
                    batch.clear()

        if batch:
            connection.executemany(upsert, batch)
        connection.commit()

        side_stats: dict[str, dict[str, object]] = {}
        for side in ("S2", "S3"):
            appearing, unique, multiple, side_max = connection.execute(
                "SELECT COUNT(*), "
                "SUM(CASE WHEN s1_count = 1 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN s1_count > 1 THEN 1 ELSE 0 END), "
                "MAX(s1_count) FROM side_counts WHERE side = ?",
                (side,),
            ).fetchone()
            appearing = int(appearing or 0)
            unique = int(unique or 0)
            multiple = int(multiple or 0)
            side_stats[side] = {
                "appearing": appearing,
                "unique": unique,
                "multiple": multiple,
                "maximum": int(side_max or 0),
                "unique_percentage": _percentage(unique, appearing),
            }
        connection.close()

    report.heading("GROUND TRUTH ANALYSIS")
    report.add(f"Path: {GROUND_TRUTH}")
    report.add(f"Columns: {', '.join(GROUND_TRUTH_COLUMNS)}")
    report.add("Required columns present: yes")
    report.add(f"Total Source 1 entities: {total_entities:,}")
    report.add(f"Total positive links: {total_links:,}")
    report.add(f"Average matches per Source 1: {total_links / total_entities if total_entities else 0:.4f}")
    report.add(f"Median matches per Source 1: {_median_from_counts(cardinalities, total_entities):.2f}")
    report.add(f"Maximum matches for one Source 1: {maximum:,}")
    report.add("Cardinality distribution:")
    report.add(f"  0 matches: {cardinalities[0]:,}")
    report.add(f"  Exactly 1 match: {cardinalities[1]:,}")
    report.add(f"  Exactly 2 matches: {cardinalities[2]:,}")
    report.add(f"  Exactly 3 matches: {cardinalities[3]:,}")
    report.add(f"  4+ matches: {sum(count for size, count in cardinalities.items() if size >= 4):,}")
    report.add(f"Matched IDs with an unexpected source prefix: {unknown_prefix_links:,}")

    report.subheading("S2/S3 side uniqueness")
    for side in ("S2", "S3"):
        stats = side_stats[side]
        report.add(f"{side} IDs appearing as a match: {stats['appearing']:,}")
        report.add(f"{side} IDs matched to exactly one S1: {stats['unique']:,}")
        report.add(f"{side} IDs matched to multiple S1s: {stats['multiple']:,}")
        report.add(f"Maximum S1s linked to one {side} ID: {stats['maximum']:,}")
        report.add(f"{side}-side uniqueness percentage: {stats['unique_percentage']:.4f}%")

    return {
        "total_entities": total_entities,
        "singleton_count": cardinalities[0],
        "positive_samples": positive_samples,
        "singleton_samples": singleton_samples,
        "side_stats": side_stats,
    }


def _sample_text(details: dict[str, dict[str, str]], entity_id: str) -> tuple[str, str]:
    record = details.get(entity_id, {})
    return record.get("name", "<NOT FOUND>"), record.get("address", "<NOT FOUND>")


def add_sample_analysis(
    report: TextReport,
    ground_truth_stats: dict[str, object],
    details: dict[str, dict[str, str]],
) -> None:
    report.heading("GROUND TRUTH SAMPLE ANALYSIS")
    report.add("Positive S1 -> S2/S3 examples:")
    for index, (source1_id, matched_id) in enumerate(
        ground_truth_stats["positive_samples"], start=1
    ):
        source_name, source_address = _sample_text(details, source1_id)
        matched_name, matched_address = _sample_text(details, matched_id)
        report.add(f"Example {index}")
        report.add(f"  S1: {source1_id} | name={source_name} | address={source_address}")
        report.add(f"  Match: {matched_id} | name={matched_name} | address={matched_address}")

    report.add("")
    report.add("Singleton examples (ground truth has no matches):")
    for source1_id in ground_truth_stats["singleton_samples"]:
        source_name, source_address = _sample_text(details, source1_id)
        report.add(f"  {source1_id} | name={source_name} | address={source_address}")


def main() -> None:
    report = TextReport()
    report.heading("PHASE 1: EDA / DATA FORENSICS")
    report.add("Raw data was analyzed without modifying or removing any rows.")
    report.add("Duplicate counts mean duplicate rows after the first occurrence.")

    ground_truth_stats = analyze_ground_truth(report)
    positive_samples = ground_truth_stats["positive_samples"]
    singleton_samples = ground_truth_stats["singleton_samples"]
    source1_sample_ids = {source1_id for source1_id, _ in positive_samples}
    source1_sample_ids.update(singleton_samples)
    source2_sample_ids = {matched_id for _, matched_id in positive_samples if matched_id.startswith("S2-")}
    source3_sample_ids = {matched_id for _, matched_id in positive_samples if matched_id.startswith("S3-")}

    report.heading("TRAINING DATASET OVERVIEW AND FORENSICS")
    train_summaries: dict[str, dict[str, object]] = {}
    all_details: dict[str, dict[str, str]] = {}
    for label, path, sample_ids in (
        ("Train S1", TRAIN_SOURCE1, source1_sample_ids),
        ("Train S2", TRAIN_SOURCE2, source2_sample_ids),
        ("Train S3", TRAIN_SOURCE3, source3_sample_ids),
    ):
        summary, details = analyze_source(label, path, report, sample_ids, True)
        train_summaries[label] = summary
        all_details.update(details)

    add_sample_analysis(report, ground_truth_stats, all_details)

    report.heading("TEST DATA BASIC CHECK")
    for label, path in (
        ("Test S1", TEST_SOURCE1),
        ("Test S2", TEST_SOURCE2),
        ("Test S3", TEST_SOURCE3),
    ):
        analyze_source(label, path, report, set(), False)

    report_path = PROJECT_ROOT / "reports" / "phase1_eda.txt"
    report.write(report_path)

    s1 = train_summaries["Train S1"]
    side_stats = ground_truth_stats["side_stats"]
    total_entities = int(ground_truth_stats["total_entities"])
    singleton_count = int(ground_truth_stats["singleton_count"])
    print("\nPHASE 1 EDA COMPLETE", flush=True)
    print(f"S1 rows: {train_summaries['Train S1']['rows']:,}", flush=True)
    print(f"S2 rows: {train_summaries['Train S2']['rows']:,}", flush=True)
    print(f"S3 rows: {train_summaries['Train S3']['rows']:,}", flush=True)
    print(f"S1 missing names: {s1['missing_name']:,}", flush=True)
    print(f"S1 missing addresses: {s1['missing_address']:,}", flush=True)
    print(f"Singleton ratio: {_percentage(singleton_count, total_entities):.4f}%", flush=True)
    print(f"S2-side uniqueness: {side_stats['S2']['unique_percentage']:.4f}%", flush=True)
    print(f"S3-side uniqueness: {side_stats['S3']['unique_percentage']:.4f}%", flush=True)
    print(f"Report saved to:\n{report_path}", flush=True)


if __name__ == "__main__":
    main()
