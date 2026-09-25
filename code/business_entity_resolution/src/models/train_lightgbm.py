from pathlib import Path
import json
import lightgbm as lgb
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "training_pairs_hard_negative.tsv"
)

OUTPUT_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = OUTPUT_DIR / "lightgbm_entity_match.txt"
FEATURE_FILE = OUTPUT_DIR / "lightgbm_features.json"


FEATURE_COLUMNS = [
    "name_norm_exact",
    "name_compact_exact",
    "name_no_suffix_exact",
    "name_sorted_exact",
    "name_translit_exact",
    "name_jaccard",
    "name_overlap",
    "name_token_intersection",
    "name_edit_similarity",
    "name_compact_edit_similarity",
    "name_no_suffix_edit_similarity",
    "name_translit_edit_similarity",
    "name_sorted_edit_similarity",
    "name_length_ratio",
    "name_length_diff",
    "name_token_count_diff",
    "address_norm_exact",
    "address_compact_exact",
    "address_sorted_exact",
    "address_translit_exact",
    "address_jaccard",
    "address_overlap",
    "address_token_intersection",
    "address_edit_similarity",
    "address_compact_edit_similarity",
    "address_numeric_jaccard",
    "address_numeric_overlap",
    "address_numeric_exact",
    "address_length_ratio",
    "address_length_diff",
    "address_token_count_diff",
    "country_exact",
    "country_missing_either",
]


def main():

    print("=" * 70)
    print("LIGHTGBM ENTITY MATCH MODEL")
    print("=" * 70)

    print("\nLoading training data...")

    df = pd.read_csv(
        TRAIN_FILE,
        sep="\t",
        low_memory=False,
    )

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------

    required = set(FEATURE_COLUMNS + ["label"])

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # Remove accidental duplicate columns if any
    # ---------------------------------------------------------

    df = df.loc[:, ~df.columns.duplicated()]

    # ---------------------------------------------------------
    # Prepare X / y
    # ---------------------------------------------------------

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(np.int8)

    print("\nClass distribution:")
    print(f"Positive: {int((y == 1).sum()):,}")
    print(f"Negative: {int((y == 0).sum()):,}")
    print(f"Positive rate: {y.mean():.6f}")

    # ---------------------------------------------------------
    # Entity-level weighting
    # ---------------------------------------------------------
    #
    # We already created hard negatives.
    # Do not randomly split pairs here.
    #
    # Training data contains only training S1 entities.
    # Validation entities are completely separate.
    #
    # We use scale_pos_weight to compensate for class imbalance.
    # ---------------------------------------------------------

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    scale_pos_weight = negatives / positives

    print(
        f"\nscale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # ---------------------------------------------------------
    # LightGBM model
    # ---------------------------------------------------------

    model = lgb.LGBMClassifier(
        objective="binary",

        n_estimators=600,

        learning_rate=0.04,

        num_leaves=31,

        max_depth=-1,

        min_child_samples=50,

        subsample=0.85,

        colsample_bytree=0.85,

        reg_alpha=0.5,
        reg_lambda=1.0,

        scale_pos_weight=scale_pos_weight,

        random_state=42,

        n_jobs=-1,

        verbosity=-1,
    )

    print("\nTraining LightGBM...")

    model.fit(
        X,
        y,
    )

    print("Training complete.")

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print("\nFeature importance:")
    print(
        importance.to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    booster = model.booster_

    booster.save_model(
        str(MODEL_FILE)
    )

    print(
        f"\nModel saved to:\n"
        f"{MODEL_FILE}"
    )

    # ---------------------------------------------------------
    # Save feature schema
    # ---------------------------------------------------------

    with open(
        FEATURE_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            FEATURE_COLUMNS,
            f,
            indent=2,
        )

    print(
        f"Feature schema saved to:\n"
        f"{FEATURE_FILE}"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()