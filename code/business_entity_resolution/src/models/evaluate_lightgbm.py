from pathlib import Path
import json

import lightgbm as lgb
import pandas as pd

from src.evaluation.scorer import macro_f05


PROJECT_ROOT = Path(__file__).resolve().parents[4]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_entity_match.txt"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_features.json"
)

VALIDATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "validation_pairs_features_sample.tsv"
)

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "reports"
    / "splits"
    / "validation_ground_truth.tsv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "lightgbm_validation_predictions.tsv"
)


def load_ground_truth(entity_ids):
    df = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        dtype=str,
    )

    entity_ids = set(entity_ids)

    truth = {}

    for _, row in df.iterrows():

        s1 = row["source1_entity_id"]

        if s1 not in entity_ids:
            continue

        value = row["matched_entity_ids"]

        if pd.isna(value) or str(value).strip() == "":
            truth[s1] = set()
        else:
            truth[s1] = set(
                str(value).split("|")
            )

    return truth

def build_predictions(
    df,
    entity_ids,
    threshold,
    top_k=None,
):
    predictions = {
        s1: set()
        for s1 in entity_ids
    }

    for s1, group in df.groupby(
        "source1_entity_id",
        sort=False,
    ):

        selected = group[
            group["match_probability"] >= threshold
        ]

        if top_k is not None:
            selected = (
                selected
                .sort_values(
                    "match_probability",
                    ascending=False,
                )
                .head(top_k)
            )

        predictions[s1] = set(
            selected["candidate_entity_id"].astype(str)
        )

    return predictions

def main():

    print("=" * 70)
    print("LIGHTGBM VALIDATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print("\nLoading model...")

    model = lgb.Booster(
        model_file=str(MODEL_FILE)
    )

    with open(
        FEATURE_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        feature_columns = json.load(f)

    print(
        f"Model features: {len(feature_columns)}"
    )

    # ---------------------------------------------------------
    # Load validation data
    # ---------------------------------------------------------

    print("\nLoading validation features...")

    df = pd.read_csv(
        VALIDATION_FILE,
        sep="\t",
        dtype=str,
        low_memory=False,
    )

    # Remove duplicate column names.
    df = df.loc[
        :,
        ~df.columns.duplicated()
    ]

    print(
        f"Candidate rows: {len(df):,}"
    )

    # ---------------------------------------------------------
    # Convert model features to numeric
    # ---------------------------------------------------------

    missing = [
        column
        for column in feature_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing model features:\n"
            + "\n".join(missing)
        )

    X = df[feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if X.isna().any().any():

        bad_columns = (
            X.columns[X.isna().any()]
            .tolist()
        )

        raise RuntimeError(
            "NaN/non-numeric values found in:\n"
            + "\n".join(bad_columns)
        )

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------

    print("\nRunning LightGBM predictions...")

    df["match_probability"] = model.predict(
        X
    )

    print(
        f"Probability min : "
        f"{df['match_probability'].min():.8f}"
    )

    print(
        f"Probability max : "
        f"{df['match_probability'].max():.8f}"
    )

    print(
        f"Probability mean: "
        f"{df['match_probability'].mean():.8f}"
    )

    # ---------------------------------------------------------
    # Save probabilities
    # ---------------------------------------------------------

    df[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "match_probability",
        ]
    ].to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved pair probabilities:\n"
        f"{OUTPUT_FILE}"
    )

    # ---------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------

    validation_entity_ids = df[
        "source1_entity_id"
    ].unique()

    truth = load_ground_truth(
        validation_entity_ids
    )
    print(
        f"\nValidation pilot entities: "
        f"{len(truth):,}"
    )


    # ---------------------------------------------------------
    # Threshold sweep
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP")
    print("=" * 70)

    thresholds = [
        0.01,
        0.02,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        0.95,
        0.98,
        0.99,
    ]

    results = []

    for threshold in thresholds:

        predictions = build_predictions(
            df,
            validation_entity_ids,
            threshold,
        )

        score = macro_f05(
            truth,
            predictions,
        )

        predicted_links = sum(
            len(matches)
            for matches in predictions.values()
        )

        results.append(
            (
                threshold,
                score,
                predicted_links,
            )
        )

        print(
            f"threshold={threshold:.2f} "
            f"F0.5={score:.6f} "
            f"predicted_links={predicted_links:,}"
        )

    # ---------------------------------------------------------
    # Best threshold
    # ---------------------------------------------------------

    best = max(
        results,
        key=lambda x: x[1],
    )

    print("\n" + "=" * 70)
    print("BEST THRESHOLD")
    print("=" * 70)

    print(
        f"Threshold       : {best[0]:.2f}"
    )

    print(
        f"Macro F0.5      : {best[1]:.6f}"
    )

    print(
        f"Predicted links : {best[2]:,}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()