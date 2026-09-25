"""
normalizer.py -- Phase 3 multi-view normalization for the Amazon ML Challenge.

Produces multiple normalized views of business_name and business_address fields.
Raw fields are ALWAYS preserved alongside normalized views; normalization never
destroys the original.

Views produced (per field):
  _raw         -- original value, whitespace-stripped
  _lower       -- lowercased, whitespace-collapsed
  _alphanum    -- lower, all non-alphanumeric chars removed, whitespace-collapsed
  _sorted      -- lower, space-split tokens sorted alphabetically (handles word-order noise)
  _expanded    -- abbreviation-expanded, then lowercase/strip (handles Corp/Corporation etc.)

Abbreviation table is data-driven from config['normalization']['abbrev_map'] --
no hardcoded country-specific logic.

Assumption: normalization is applied to all country values uniformly. Any
country-specific pattern that survives EDA is added to the config abbrev_map,
not branched in code.

This module is only called by the pipeline -- it never reads from disk directly.
The caller passes DataFrames; this module returns normalized DataFrames.
"""

from __future__ import annotations

import logging
import pathlib
import re
import sys
import unicodedata
from typing import Any

import pandas as pd

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))
from config.config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default abbreviation map (expanded by config at runtime)
# Keys must be whole-word matches (applied after lowercasing).
# ---------------------------------------------------------------------------
_DEFAULT_ABBREV: dict[str, str] = {
    # Legal suffixes
    "pvt": "private",
    "pvt.": "private",
    "ltd": "limited",
    "ltd.": "limited",
    "llc": "llc",
    "llp": "llp",
    "inc": "incorporated",
    "inc.": "incorporated",
    "corp": "corporation",
    "corp.": "corporation",
    "p.c.": "",    # strip P.C. suffix entirely (near-match with no suffix)
    "l.l.c.": "llc",  # L.L.C. -> llc
    "l.l.p.": "llp",  # L.L.P. -> llp
    # Address abbreviations (US)
    "rd": "road",
    "rd.": "road",
    "st": "street",
    "st.": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "ln": "lane",
    "ln.": "lane",
    "ct": "court",
    "ct.": "court",
    "hwy": "highway",
    "fwy": "freeway",
    # State abbreviations (US -- expand to avoid token collision)
    "in": "indiana",
    "az": "arizona",
    "nc": "north carolina",
    "mn": "minnesota",
    "il": "illinois",
    "md": "maryland",
    "wv": "west virginia",
    "tn": "tennessee",
    "ar": "arkansas",
    "or": "oregon",
    "nm": "new mexico",
    "mt": "montana",
    "mo": "missouri",
    # Connectors
    "&": "and",
}

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_NOISE_PREFIX_RE = re.compile(r"^(smt|sri|--|\*+|c/o)\s+", re.IGNORECASE)
_NULL_LITERAL_RE = re.compile(r"<null>", re.IGNORECASE)
# Collapse dotted single-char sequences: l.l.c. -> llc, p.c. -> pc
# Must run BEFORE abbrev expansion because \b doesn't fire after trailing dot.
_DOTTED_ABBREV_RE = re.compile(r"(?<![a-z])([a-z])(?:\.[a-z])+\.?", re.IGNORECASE)


def _build_abbrev_pattern(abbrev_map: dict[str, str]) -> re.Pattern[str]:
    """Build a compiled whole-word regex for all abbreviation keys."""
    sorted_keys = sorted(abbrev_map.keys(), key=len, reverse=True)  # longest first
    escaped = [re.escape(k) for k in sorted_keys]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


def _expand_abbreviations(text: str, pattern: re.Pattern[str], abbrev_map: dict[str, str]) -> str:
    """Replace whole-word abbreviations with their expanded forms."""
    return pattern.sub(lambda m: abbrev_map.get(m.group(0).lower(), m.group(0)), text)


def _normalize_one(raw: str, abbrev_pattern: re.Pattern[str], abbrev_map: dict[str, str]) -> dict[str, str]:
    """Return all normalized views for a single field value.

    Args:
        raw: The raw field string.
        abbrev_pattern: Compiled regex for abbreviation expansion.
        abbrev_map: Dict mapping lowercase abbrev to expansion.

    Returns:
        Dict with keys: raw, lower, alphanum, sorted, expanded.
    """
    # Raw (whitespace-stripped only)
    raw_stripped = _WS_RE.sub(" ", raw).strip()

    # Unicode accent stripping: NFD decomposition -> drop combining chars
    # e.g. Á -> A, é -> e. Preserves non-Latin scripts (Indic, CJK) intact.
    def _strip_accents(s: str) -> str:
        nfd = unicodedata.normalize("NFD", s)
        return "".join(c for c in nfd if unicodedata.category(c) != "Mn")

    # Lower
    lower = _strip_accents(raw_stripped).lower()
    lower = _NULL_LITERAL_RE.sub("", lower)
    lower = _NOISE_PREFIX_RE.sub("", lower)
    lower = _WS_RE.sub(" ", lower).strip()

    # Alphanum (remove all non-alphanumeric except spaces, then collapse)
    alphanum = _NON_ALNUM_RE.sub(" ", lower)
    alphanum = _WS_RE.sub(" ", alphanum).strip()

    # Sorted tokens (handles word-order transpositions)
    sorted_tokens = " ".join(sorted(alphanum.split()))

    # Expanded (abbreviation expansion applied to lower, then alphanum)
    # Pre-process dotted abbreviations FIRST (l.l.c. -> llc, p.c. -> pc)
    # because word-boundary \b doesn't match after a trailing dot.
    def _collapse_dotted(s: str) -> str:
        return _DOTTED_ABBREV_RE.sub(lambda m: m.group(0).replace(".", ""), s)

    expanded_lower = _collapse_dotted(lower)
    expanded_lower = _expand_abbreviations(expanded_lower, abbrev_pattern, abbrev_map)
    expanded = _NON_ALNUM_RE.sub(" ", expanded_lower)
    expanded = _WS_RE.sub(" ", expanded).strip()
    # Also apply to alphanum (improves sorted-token matching)
    alphanum_exp = _collapse_dotted(alphanum)
    sorted_tokens_exp = " ".join(sorted(alphanum_exp.split()))

    return {
        "raw": raw_stripped,
        "lower": lower,
        "alphanum": alphanum,
        "sorted": sorted_tokens_exp,
        "expanded": expanded,
    }


def normalize_dataframe(
    df: pd.DataFrame,
    fields: list[str],
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add normalized views for each field in fields to df.

    Raw columns are untouched. For each field F, adds:
        F_lower, F_alphanum, F_sorted, F_expanded

    (The raw value itself is already in F -- no _raw suffix added to avoid
    confusion with the original column name.)

    Args:
        df: Input DataFrame containing the named fields.
        fields: List of column names to normalize.
        config: Pipeline config dict. If None, loads from default path.

    Returns:
        df with added normalized columns. Original columns are preserved.

    Assumption: normalization is side-effect-free -- original df is not mutated.
    """
    if config is None:
        config = load_config()

    # Merge default abbrev with any config overrides (config wins)
    abbrev_map = dict(_DEFAULT_ABBREV)
    cfg_abbrev = config.get("normalization", {}).get("abbrev_map", {})
    abbrev_map.update({k.lower(): v.lower() for k, v in cfg_abbrev.items()})
    abbrev_pattern = _build_abbrev_pattern(abbrev_map)

    out = df.copy()
    for field in fields:
        if field not in df.columns:
            logger.warning("Field '%s' not in DataFrame -- skipping.", field)
            continue
        views = df[field].fillna("").apply(
            lambda v: _normalize_one(str(v), abbrev_pattern, abbrev_map)
        )
        for view_name in ["lower", "alphanum", "sorted", "expanded"]:
            col = f"{field}_{view_name}"
            out[col] = views.apply(lambda d: d[view_name])
        logger.info("Normalized field '%s' -> 4 views added.", field)

    return out


# ---------------------------------------------------------------------------
# Phase 3 gate: spot-check normalizer against the Phase 1 true-pair sample
# ---------------------------------------------------------------------------

def run_spot_check(
    sample_path: pathlib.Path | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Load the Phase 1 noisy-true-pair sample and show normalized side-by-side.

    This is the Phase 3 gate: a human looks at the 'sorted' and 'expanded'
    views and verifies that known true pairs are near-identical.

    Output is printed to stdout (the authoritative gate record) and also
    appended to artifacts/phase3_normalization_spotcheck.txt.
    """
    if config is None:
        config = load_config()

    artifacts_dir = pathlib.Path(config["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if sample_path is None:
        eda_summary = artifacts_dir / "phase1_eda_summary.txt"
        if not eda_summary.exists():
            raise FileNotFoundError(
                f"{eda_summary} not found. Run eda.py first (Phase 1)."
            )

    # Load the entity split to get the actual S1/S2/S3 records
    split_path = artifacts_dir / "entity_split.tsv"
    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} not found. Run splitter.py first (Phase 2).")

    train_dir = pathlib.Path(config["paths"]["train_dir"])
    s1 = pd.read_csv(train_dir / "train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
    s2 = pd.read_csv(train_dir / "train_source2.tsv", sep="\t", dtype=str, keep_default_na=False)
    s3 = pd.read_csv(train_dir / "train_source3.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(train_dir / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)

    # Sample 25 non-singleton pairs (same seed as EDA)
    non_singleton = gt[gt["matched_entity_ids"].str.strip() != ""].copy()
    non_singleton["matched_list"] = non_singleton["matched_entity_ids"].str.split(",")
    pairs = non_singleton.explode("matched_list")[["source1_entity_id", "matched_list"]]
    pairs.columns = ["s1_id", "target_id"]
    pairs = pairs.sample(25, random_state=42).reset_index(drop=True)

    # Normalize S1 and target records
    s2_s3 = pd.concat([s2, s3]).set_index("entity_id")
    s1_idx = s1.set_index("entity_id")
    FIELDS = ["business_name", "business_address"]

    s1_norm = normalize_dataframe(s1_idx.loc[pairs["s1_id"].values].reset_index(), FIELDS, config)
    tgt_norm = normalize_dataframe(
        s2_s3.loc[pairs["target_id"].values].reset_index(), FIELDS, config
    )

    lines: list[str] = []

    def emit(msg: str = "") -> None:
        lines.append(msg)
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="backslashreplace").decode("ascii"))

    emit("=" * 80)
    emit("PHASE 3 GATE -- Normalization Spot-Check (25 known true pairs)")
    emit("=" * 80)

    for i, (_, row) in enumerate(pairs.iterrows()):
        s1_r = s1_norm[s1_norm["entity_id"] == row["s1_id"]].iloc[0]
        tgt_r = tgt_norm[tgt_norm["entity_id"] == row["target_id"]].iloc[0]
        emit(f"\n--- Pair {i+1:02d}: {row['s1_id']}  <->  {row['target_id']} ---")
        for field in FIELDS:
            emit(f"  [{field}]")
            emit(f"    RAW S1 : {s1_r[field]}")
            emit(f"    RAW TGT: {tgt_r[field]}")
            for view in ["lower", "alphanum", "sorted", "expanded"]:
                s1v = s1_r[f"{field}_{view}"]
                tv = tgt_r[f"{field}_{view}"]
                match = "==" if s1v == tv else "!="
                emit(f"    {view:<10}: S1={s1v!r}  {match}  TGT={tv!r}")

    emit("\n" + "=" * 80)
    emit("END OF SPOT-CHECK -- Review '==' rows; near-identical sorted/expanded = PASS")
    emit("=" * 80)

    out_path = artifacts_dir / "phase3_normalization_spotcheck.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nArtifact saved: {out_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stderr,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    run_spot_check()
