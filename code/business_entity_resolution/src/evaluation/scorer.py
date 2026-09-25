import pandas as pd


def f05_score(predicted, actual):
    """
    Calculate F0.5 for one Source 1 entity.
    """

    predicted = set(predicted)
    actual = set(actual)

    true_positive = len(predicted & actual)

    if len(predicted) == 0:
        precision = 1.0 if len(actual) == 0 else 0.0
    else:
        precision = true_positive / len(predicted)

    if len(actual) == 0:
        recall = 1.0 if len(predicted) == 0 else 0.0
    else:
        recall = true_positive / len(actual)

    if precision == 0 and recall == 0:
        return 0.0

    return (1.25 * precision * recall) / (
        0.25 * precision + recall
    )


def _load_entity_matches(path):
    """Load a two-column entity-match TSV into a dictionary."""

    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    entity_matches = {}

    for source1_id, matches in df[
        ["source1_entity_id", "matched_entity_ids"]
    ].itertuples(index=False, name=None):
        entity_matches[source1_id] = [
            entity_id.strip()
            for entity_id in matches.split(",")
            if entity_id.strip()
        ]

    return entity_matches


def load_ground_truth(path):
    """
    Load train_ground_truth.tsv.

    Returns:
        Dictionary:
        {
            "S1-00001": ["S2-00047", "S3-00012"],
            ...
        }
    """

    return _load_entity_matches(path)


def load_predictions(path):
    """
    Load matching_results.tsv.
    """

    return _load_entity_matches(path)


def macro_f05(ground_truth, predictions):
    """
    Calculate macro F0.5 across Source 1 entities.
    """

    scores = []

    for source1_id, actual_matches in ground_truth.items():

        predicted_matches = predictions.get(source1_id, [])

        score = f05_score(
            predicted_matches,
            actual_matches
        )

        scores.append(score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def evaluate(ground_truth_path, prediction_path):
    """Load ground truth and predictions, then return their macro F0.5."""

    ground_truth = load_ground_truth(ground_truth_path)
    predictions = load_predictions(prediction_path)
    return macro_f05(ground_truth, predictions)


if __name__ == "__main__":
    test_cases = [
        (
            "official worked example",
            ["S2-00047", "S2-00193", "S3-00812"],
            ["S2-00047", "S3-00812"],
            0.7142857,
        ),
        ("empty singleton", [], [], 1.0),
        ("false positive singleton", ["S2-001"], [], 0.0),
        ("missed match", [], ["S2-001"], 0.0),
        (
            "perfect match",
            ["S2-001", "S3-002"],
            ["S2-001", "S3-002"],
            1.0,
        ),
    ]

    all_passed = True
    for name, predicted, actual, expected in test_cases:
        result = f05_score(predicted, actual)
        passed = abs(result - expected) < 1e-7
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name} (expected={expected:.7f}, got={result:.7f})")

    print(f"{'PASS' if all_passed else 'FAIL'}: Phase 0 scorer tests")
