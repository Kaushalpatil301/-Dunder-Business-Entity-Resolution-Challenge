"""cleanup_and_launch.py — clears stale artifacts, then launches master_run.py"""
import pathlib, subprocess, sys, shutil, os

REPO = pathlib.Path(__file__).resolve().parent
ARTS = REPO / "artifacts"

# Remove stale blocking outputs
stale = [
    ARTS / "candidate_pairs_all_folds_flat.tsv",
    ARTS / "_tmp_channels",
    ARTS / "_tmp_test_channels",
]
for p in stale:
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
        print(f"Removed dir: {p}")
    elif p.exists():
        p.unlink()
        print(f"Removed: {p}")

# Also remove feature/model files so everything reruns fresh
for pattern in ["features_fold_*.tsv", "lgbm_model.txt",
                "threshold.txt", "calibrator.pkl",
                "phase4_recall_report.txt"]:
    for f in ARTS.glob(pattern):
        f.unlink()
        print(f"Removed: {f}")

print("Stale artifacts cleared. Launching master_run.py ...")
result = subprocess.run(
    [sys.executable, "master_run.py"],
    cwd=str(REPO),
)
sys.exit(result.returncode)
