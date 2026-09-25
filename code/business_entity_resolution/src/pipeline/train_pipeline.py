"""
train_pipeline.py -- Phase 5-7 orchestrator: train the full pipeline.

Steps:
  1. Run blocking (fold A + all-folds for features, fold D for recall gate)
  2. Extract pairwise features for fold A (train), fold B (tune), fold C (calibrate)
  3. Train LightGBM on fold A, tune threshold on fold B
  4. Fit isotonic calibration on fold C
  5. Report Phase 4 recall gate from fold D artifact

All intermediate artifacts are saved under artifacts/ for reproducibility.

Usage:
  python src/pipeline/train_pipeline.py
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parents[5]
_SRC  = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SRC))

from config.config import load_config

logger = logging.getLogger(__name__)


def _py(script_rel: str, *args: str) -> None:
    """Run a src/ Python script as a subprocess."""
    script = _SRC / script_rel
    cmd = [sys.executable, str(script)] + list(args)
    logger.info("Running: %s", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_rel}")


def run_train_pipeline() -> None:
    cfg  = load_config()
    arts = pathlib.Path(cfg["paths"]["artifacts_dir"])
    trn  = pathlib.Path(cfg["paths"]["train_dir"])
    arts.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("TRAIN PIPELINE STARTED")
    logger.info("=" * 70)

    # -----------------------------------------------------------------
    # STEP 1: Phase 4 — Blocking union (recall gate on fold D)
    # -----------------------------------------------------------------
    logger.info("\n[STEP 1] Phase 4: Blocking union — fold D recall gate")
    _py("blocking/union.py")   # default: fold-d-only mode

    # -----------------------------------------------------------------
    # STEP 2: Phase 4 continued — Generate candidates for ALL training folds
    # (Needed to build feature matrices for A, B, C)
    # -----------------------------------------------------------------
    logger.info("\n[STEP 2] Phase 4: Generate candidates for all train folds")
    _py("blocking/union.py", "--all-folds")

    # -----------------------------------------------------------------
    # STEP 3: Phase 5 — Feature extraction for folds A, B, C
    # -----------------------------------------------------------------
    split_file = arts / "entity_split.tsv"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Entity split not found: {split_file}. Run src/data/splitter.py first."
        )

    import pandas as pd
    split_df = pd.read_csv(split_file, sep="\t", dtype=str)
    cands_all = arts / "candidate_pairs_v1.tsv"

    for fold in ("A", "B", "C"):
        logger.info("\n[STEP 3.%s] Phase 5: Features for fold %s", fold, fold)
        fold_ids = set(split_df[split_df["fold"] == fold]["source1_entity_id"])

        # Filter candidates to this fold's S1 entities
        cands_full = pd.read_csv(cands_all, sep="\t", dtype=str, keep_default_na=False)
        col_s1 = "source1_entity_id" if "source1_entity_id" in cands_full.columns else "s1_id"
        fold_cands = cands_full[cands_full[col_s1].isin(fold_ids)]
        fold_cands_path = arts / f"candidate_pairs_fold_{fold.lower()}.tsv"
        fold_cands.to_csv(fold_cands_path, sep="\t", index=False, encoding="utf-8")
        logger.info("  Fold %s candidates: %d rows → %s", fold, len(fold_cands), fold_cands_path)

        _py(
            "features/pairwise.py",
            "--candidates", str(fold_cands_path),
            "--input-dir",  str(trn),
            "--gt",         str(trn / "train_ground_truth.tsv"),
            "--output",     str(arts / f"features_fold_{fold.lower()}.tsv"),
        )

    # -----------------------------------------------------------------
    # STEP 4: Phase 6 — Train LightGBM, tune threshold on fold B
    # -----------------------------------------------------------------
    logger.info("\n[STEP 4] Phase 6: Train LightGBM")
    _py("models/matcher.py", "--train")

    # -----------------------------------------------------------------
    # STEP 5: Phase 7 — Fit isotonic calibration on fold C
    # -----------------------------------------------------------------
    logger.info("\n[STEP 5] Phase 7: Fit calibrator on fold C")
    _py("calibration/calibrator.py", "--fit",
        "--features", str(arts / "features_fold_c.tsv"))

    logger.info("\n" + "=" * 70)
    logger.info("TRAIN PIPELINE COMPLETE")
    logger.info("  Model:       %s", arts / "lgbm_model.txt")
    logger.info("  Threshold:   %s", arts / "threshold.txt")
    logger.info("  Calibrator:  %s", arts / "calibrator.pkl")
    logger.info("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    run_train_pipeline()
