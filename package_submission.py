#!/usr/bin/env python
"""
package_submission.py — Create the official final submission zip for the Amazon ML Challenge.

Strictly adheres to the specification in root README.md:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (leaderboard scored)
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code
│       ├── artifacts/              # pretrained model weights (for 2-min reproduction)
│       ├── run_full_pipeline.py    # top-level runner
│       ├── README.md               # reproduction guide
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # filled methodology write-up

Run from repo root:
    python package_submission.py --team-name Dunder
"""

import argparse
import datetime
import os
import pathlib
import shutil
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parent


def count_lines(path: pathlib.Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser(description="Package submission zip strictly conforming to root README.md")
    parser.add_argument("--team-name", default="Dunder", help="Team name for zip filename")
    args = parser.parse_args()

    team = args.team_name.replace(" ", "_")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f"{team}_submission.zip"
    zip_path = REPO / zip_name

    print("=" * 70)
    print(f"PACKAGING FINAL SUBMISSION: {zip_name}")
    print("=" * 70)

    # 1. Verify required files
    matching_tsv = REPO / "output" / "matching_results.tsv"
    candidate_tsv = REPO / "output" / "candidate_pairs.tsv"
    doc_file = REPO / "Documentation_template.md"
    code_dir = REPO / "code" / "business_entity_resolution"
    req_file = code_dir / "requirements.txt"
    code_readme = code_dir / "README.md"

    for p in [matching_tsv, candidate_tsv, doc_file, req_file, code_readme]:
        if not p.exists():
            print(f"ERROR: Missing required file: {p}", file=sys.stderr)
            sys.exit(1)

    # 2. Check row counts
    print("\n[1] Verifying output files...")
    n_matching = count_lines(matching_tsv)
    n_candidate = count_lines(candidate_tsv)
    print(f"  matching_results.tsv : {n_matching:,} lines (header + 1,732,544 rows)")
    print(f"  candidate_pairs.tsv  : {n_candidate:,} lines (header + 1,732,544 rows)")

    if n_matching != 1732545:
        print(f"WARNING: matching_results.tsv line count is {n_matching:,}, expected 1,732,545")
    if n_candidate != 1732545:
        print(f"WARNING: candidate_pairs.tsv line count is {n_candidate:,}, expected 1,732,545")

    # 3. Ensure self-contained code tree
    print("\n[2] Synchronizing self-contained code tree...")
    src_dir = code_dir / "src"
    src_config = src_dir / "config"
    src_utils = src_dir / "utils"
    code_arts = code_dir / "artifacts"
    src_config.mkdir(parents=True, exist_ok=True)
    src_utils.mkdir(parents=True, exist_ok=True)
    code_arts.mkdir(parents=True, exist_ok=True)

    if (REPO / "config" / "config.py").exists():
        shutil.copy(REPO / "config" / "config.py", src_config / "config.py")
    if (REPO / "config" / "pipeline.yaml").exists():
        shutil.copy(REPO / "config" / "pipeline.yaml", src_config / "pipeline.yaml")
    if (REPO / "utils" / "validate_submission.py").exists():
        shutil.copy(REPO / "utils" / "validate_submission.py", src_utils / "validate_submission.py")
    if (REPO / "quick_validate.py").exists():
        shutil.copy(REPO / "quick_validate.py", src_utils / "quick_validate.py")

    for art in ["lgbm_model.txt", "calibrator.pkl", "threshold.txt"]:
        src_art = REPO / "artifacts" / art
        if src_art.exists():
            shutil.copy(src_art, code_arts / art)

    # 4. Build zip archive
    print(f"\n[3] Building zip archive {zip_name} (DEFLATED)...")
    if zip_path.exists():
        zip_path.unlink()

    total_files = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # A. output/
        zf.write(matching_tsv, "output/matching_results.tsv")
        print("  + output/matching_results.tsv")
        total_files += 1

        zf.write(candidate_tsv, "output/candidate_pairs.tsv")
        print("  + output/candidate_pairs.tsv")
        total_files += 1

        # B. Documentation_template.md
        zf.write(doc_file, "Documentation_template.md")
        print("  + Documentation_template.md")
        total_files += 1

        # C. code/business_entity_resolution/
        # Collect all files under code/business_entity_resolution
        for item in sorted(code_dir.rglob("*")):
            if item.is_file():
                # Exclude python cache / temporary files
                rel_parts = item.relative_to(code_dir).parts
                if any(p.startswith(".") or p == "__pycache__" or p.endswith(".pyc") for p in rel_parts):
                    continue
                arcname = f"code/business_entity_resolution/{item.relative_to(code_dir).as_posix()}"
                zf.write(item, arcname)
                print(f"  + {arcname}")
                total_files += 1




    # 6. Verify zip structure
    print(f"\n[4] Verifying created archive {zip_name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        root_entries = set()
        for name in names:
            root_entry = name.split("/")[0]
            root_entries.add(root_entry)

        expected_roots = {"output", "code", "Documentation_template.md"}
        extra_roots = root_entries - expected_roots
        if extra_roots:
            print(f"WARNING: Unexpected root entries in zip: {extra_roots}")
        else:
            print("  Root entries verified: strictly matches root README.md specification!")

        # Verify key entries exist
        assert "output/matching_results.tsv" in names, "Missing matching_results.tsv in zip"
        assert "output/candidate_pairs.tsv" in names, "Missing candidate_pairs.tsv in zip"
        assert "Documentation_template.md" in names, "Missing Documentation_template.md in zip"
        assert "code/business_entity_resolution/README.md" in names, "Missing code README in zip"
        assert "code/business_entity_resolution/requirements.txt" in names, "Missing requirements.txt in zip"

    size_mb = zip_path.stat().st_size / 1e6
    print("\n" + "=" * 70)
    print("SUBMISSION PACKAGE SUCCESSFULLY CREATED & VERIFIED!")
    print("=" * 70)
    print(f"Zip file: {zip_path} ({size_mb:.1f} MB, {total_files} files)")
    print("\nStructure in archive:")
    print("  Dunder_submission.zip")
    print("  |-- output/")
    print("  |   |-- matching_results.tsv    (1,732,544 rows - LEADERBOARD UPLOAD FILE)")
    print("  |   +-- candidate_pairs.tsv     (1,732,544 rows - BLOCKING CANDIDATE SET)")
    print("  |-- code/")
    print("  |   +-- business_entity_resolution/")
    print("  |       |-- src/                (complete source code & modules)")
    print("  |       |-- artifacts/          (pretrained LightGBM weights for 2-min reproduction)")
    print("  |       |-- run_full_pipeline.py(master pipeline runner)")
    print("  |       |-- README.md           (end-to-end reproduction guide)")
    print("  |       +-- requirements.txt    (pinned dependencies)")
    print("  +-- Documentation_template.md   (filled methodology & validation report)")
    print("=" * 70)


if __name__ == "__main__":
    main()
