"""
fast_normalizer.py -- Vectorized normalization for blocking (Phase 4).

Views produced:
  name_lower            -- lowercase + whitespace collapse
  name_alphanum         -- lower + non-alphanumeric stripped
  name_sorted           -- sorted alphanum tokens (word-order invariant)
  name_expanded         -- alphanum + abbreviation expansion (pvt->private, ltd->limited, etc.)
  name_expanded_sorted  -- sorted expanded tokens (handles ltd/limited AND word-order)
  addr_alphanum         -- address alphanum
  addr_numeric          -- first digit sequence in address

All operations are vectorized pandas str ops (no Python-level .apply on large frames
except for the token-sort step, which cannot be avoided).

Assumption: input DataFrame has business_name, business_address, country columns.
"""

from __future__ import annotations

import pandas as pd


def normalize_for_blocking(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with added normalized view columns."""
    out = df.copy()

    # --- Name: lower + noise strip ---
    name = df["business_name"].fillna("").str.lower().str.strip()
    name = name.str.replace(r"<null>", "", regex=True)
    name = name.str.replace(r"^(smt|sri|--|[*]+)\s+", "", regex=True)
    name = name.str.replace(r"\s+", " ", regex=True).str.strip()
    out["name_lower"] = name

    # --- Name: alphanum ---
    alnum = name.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    alnum = alnum.str.replace(r"\s+", " ", regex=True).str.strip()
    out["name_alphanum"] = alnum

    # --- Name: sorted alphanum tokens ---
    out["name_sorted"] = alnum.str.split().apply(
        lambda toks: " ".join(sorted(toks)) if toks else ""
    )

    # --- Name: abbreviation expansion (vectorized) ---
    # Strategy: first normalise long forms -> canonical short, then expand short -> canonical long.
    # This way pvt/private/Private all reach the same expanded form.
    # Order: apply longest patterns first to avoid partial substitution.
    exp = alnum
    # Step 1: collapse any long-form variants that may already be in the data
    exp = exp.str.replace(r"\bprivate\b",       "pvt",   regex=True)
    exp = exp.str.replace(r"\blimited\b",        "ltd",   regex=True)
    exp = exp.str.replace(r"\bincorporated\b",   "inc",   regex=True)
    exp = exp.str.replace(r"\bcorporation\b",    "corp",  regex=True)
    exp = exp.str.replace(r"\bservices\b",       "svcs",  regex=True)
    exp = exp.str.replace(r"\bservice\b",        "svc",   regex=True)
    exp = exp.str.replace(r"\bmanagement\b",     "mgmt",  regex=True)
    exp = exp.str.replace(r"\binternational\b",  "intl",  regex=True)
    exp = exp.str.replace(r"\bassociates\b",     "assoc", regex=True)
    # Step 2: expand all canonical short forms -> canonical long forms
    exp = exp.str.replace(r"\bpvt\b",   "private",       regex=True)
    exp = exp.str.replace(r"\bltd\b",   "limited",       regex=True)
    exp = exp.str.replace(r"\binc\b",   "incorporated",  regex=True)
    exp = exp.str.replace(r"\bcorp\b",  "corporation",   regex=True)
    exp = exp.str.replace(r"\bsvcs\b",  "services",      regex=True)
    exp = exp.str.replace(r"\bsvc\b",   "service",       regex=True)
    exp = exp.str.replace(r"\bmgmt\b",  "management",    regex=True)
    exp = exp.str.replace(r"\bintl\b",  "international", regex=True)
    exp = exp.str.replace(r"\bassoc\b", "associates",    regex=True)
    exp = exp.str.replace(r"\s+", " ", regex=True).str.strip()
    out["name_expanded"] = exp

    out["name_expanded_sorted"] = exp.str.split().apply(
        lambda toks: " ".join(sorted(toks)) if toks else ""
    )

    # --- Address ---
    addr = df["business_address"].fillna("").str.lower().str.strip()
    addr_alnum = addr.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    addr_alnum = addr_alnum.str.replace(r"\s+", " ", regex=True).str.strip()
    out["addr_alphanum"] = addr_alnum
    out["addr_numeric"] = addr_alnum.str.extract(r"(\d+)", expand=False).fillna("")

    return out
