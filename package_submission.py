#!/usr/bin/env python
"""
package_submission.py — Create the submission zip for the Amazon ML Challenge.

Creates: <team_name>_submission.zip with:
  output/
    matching_results.tsv
    candidate_pairs.tsv
  code/business_entity_resolution/
    src/ (all source code)
    README.md
    requirements.txt
  Documentation_template.md

Run from repo root:
    python package_submission.py --team-name Dunder
"""

import argparse
import pathlib
import sys
import zipfile
import datetime

REPO = pathlib.Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-name", default="Dunder", help="Team name for zip filename")
    args = parser.parse_args()

    team = args.team_name.replace(" ", "_")
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f"{team}_submission_{ts}.zip"
    zip_path = REPO / zip_name

    print(f"Creating {zip_path} ...")

    required = [
        REPO / "output" / "matching_results.tsv",
        REPO / "output" / "candidate_pairs.tsv",
        REPO / "Documentation_template.md",
    ]
    for p in required:
        if not p.exists():
            print(f"ERROR: Required file missing: {p}", file=sys.stderr)
            sys.exit(1)

    # Check output files are non-empty (not just headers)
    for p in required[:2]:
        lines = sum(1 for _ in open(p, encoding="utf-8"))
        if lines < 2:
            print(f"ERROR: {p.name} is empty (only header)", file=sys.stderr)
            sys.exit(1)
        print(f"  {p.name}: {lines:,} rows")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Output files
        for p in [REPO / "output" / "matching_results.tsv",
                  REPO / "output" / "candidate_pairs.tsv"]:
            zf.write(p, f"output/{p.name}")
            print(f"  + output/{p.name}")

        # Code
        code_root = REPO / "code" / "business_entity_resolution"
        for src_file in sorted(code_root.rglob("*.py")):
            arcname = f"code/business_entity_resolution/{src_file.relative_to(code_root)}"
            zf.write(src_file, arcname)
        zf.write(code_root / "README.md",
                 "code/business_entity_resolution/README.md")
        zf.write(code_root / "requirements.txt",
                 "code/business_entity_resolution/requirements.txt")
        print(f"  + code/business_entity_resolution/ (all .py files + README + requirements)")

        # Root pipeline scripts
        for script in ["run_full_pipeline.py", "master_run.py", "add_new_channels.py",
                       "run_after_blocking.py", "quick_validate.py"]:
            if (REPO / script).exists():
                zf.write(REPO / script, script)
                print(f"  + {script}")

        # Documentation
        zf.write(REPO / "Documentation_template.md", "Documentation_template.md")
        print(f"  + Documentation_template.md")

        # Config
        for cfg in (REPO / "config").rglob("*.yaml"):
            zf.write(cfg, f"config/{cfg.name}")
        for cfg in (REPO / "config").rglob("*.py"):
            zf.write(cfg, f"config/{cfg.name}")
        print(f"  + config/")

        # Utils
        for u in (REPO / "utils").rglob("*.py"):
            zf.write(u, f"utils/{u.name}")
        print(f"  + utils/")

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\n✅ Done: {zip_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"\n📤 Upload output/matching_results.tsv to the leaderboard portal.")


if __name__ == "__main__":
    main()
