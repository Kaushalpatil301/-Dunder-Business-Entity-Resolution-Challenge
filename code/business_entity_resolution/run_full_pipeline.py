#!/usr/bin/env python
"""
run_full_pipeline.py — Master High-Performance Entity Resolution Pipeline.
Reproduces candidate_pairs.tsv and matching_results.tsv for Dunder Business Entity Resolution Challenge.
"""
import sys
import pathlib

CUR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CUR))
sys.path.insert(0, str(CUR / "src"))

from src.run_pipeline import main

if __name__ == "__main__":
    main()
