"""
scorer.py -- Macro-averaged F0.5 for the Amazon ML Challenge 2026.

Metric definition (from the problem statement):
    F_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    beta = 0.5  ->  F0.5 weights precision twice as much as recall.

Per-entity F0.5:
    precision_i = |pred_i intersect gt_i| / |pred_i|  (0 if pred is empty)
    recall_i    = |pred_i intersect gt_i| / |gt_i|    (1 if gt empty AND pred empty,
                                                        0 if gt empty AND pred non-empty)
    f05_i       = F0.5(precision_i, recall_i)          (0 if denominator = 0)

Macro F0.5 = mean(f05_i) over all S1 entities (including singletons).

Singleton entity: empty ground-truth match set.
  - Correct abstain (pred also empty): P=1, R=1, F0.5=1.
  - False positive (pred non-empty):   P=0, R=0, F0.5=0.

This module is read-only with respect to the model.
"""

from __future__ import annotations

import logging
from typing import Dict, FrozenSet, List, Tuple

logger = logging.getLogger(__name__)

EntityId = str
MatchSet = FrozenSet[EntityId]
GroundTruth = Dict[EntityId, MatchSet]
Predictions = Dict[EntityId, MatchSet]

BETA: float = 0.5
BETA_SQ: float = BETA ** 2


def f05_score(precision: float, recall: float) -> float:
    """Return F0.5 for a single (precision, recall) pair.

    Returns 0.0 when denominator is zero (both P and R are 0).
    """
    denom = BETA_SQ * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + BETA_SQ) * precision * recall / denom


def entity_f05(
    gt: MatchSet,
    pred: MatchSet,
) -> Tuple[float, float, float]:
    """Compute per-entity precision, recall, and F0.5.

    Singleton handling (gt is empty):
      - pred empty  -> P=1, R=1, F0.5=1  (correct abstain)
      - pred non-empty -> P=0, R=0, F0.5=0 (false positive on singleton)
    Normal entity: standard intersection ratios.

    Returns (precision, recall, f05).
    """
    if not gt:
        if not pred:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    tp = len(gt & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gt)
    return precision, recall, f05_score(precision, recall)


def macro_f05(
    ground_truth: GroundTruth,
    predictions: Predictions,
) -> float:
    """Compute macro-averaged F0.5 over all S1 entities in ground_truth.

    Every S1 entity in ground_truth must appear in predictions; a missing
    entry is treated as empty prediction and logged as a warning.

    Assumption: ground_truth contains every S1 entity including singletons
    (singletons map to an empty frozenset).

    Args:
        ground_truth: Mapping from S1 entity_id to frozenset of correct
            S2/S3 match IDs (empty frozenset for singletons).
        predictions: Mapping from S1 entity_id to frozenset of predicted
            S2/S3 match IDs (empty frozenset for abstain / singleton).

    Returns:
        Macro-averaged F0.5 score in [0, 1].
    """
    if not ground_truth:
        raise ValueError("ground_truth is empty -- nothing to score.")

    scores: List[float] = []
    for s1_id, gt_matches in ground_truth.items():
        if s1_id not in predictions:
            logger.warning(
                "S1 entity %s missing from predictions; treating as empty.", s1_id
            )
        pred_matches = predictions.get(s1_id, frozenset())
        _, _, score = entity_f05(gt_matches, pred_matches)
        scores.append(score)

    result = sum(scores) / len(scores)
    logger.info(
        "macro_f05 = %.6f over %d entities (mean of per-entity F0.5).",
        result,
        len(scores),
    )
    return result


def _run_self_test() -> None:
    """Run the PS worked example and print results for Phase 0 gate evidence.

    PS example (Section on scoring):
        S1-00001 ground truth: {S2-00001, S3-00001}  (2 matches)
        S1-00001 prediction:   {S2-00001, S3-00001, S2-00002}  (3 predicted)
        -> precision = 2/3 = 0.6667, recall = 2/2 = 1.0
        -> F0.5 = 1.25 * 0.6667 * 1.0 / (0.25 * 0.6667 + 1.0)
                = 0.8333 / 1.1667
                = 0.7143
    Macro F0.5 with 1 entity = that entity's F0.5 = 0.714.
    """
    gt: GroundTruth = {
        "S1-00001": frozenset({"S2-00001", "S3-00001"}),
    }
    pred: Predictions = {
        "S1-00001": frozenset({"S2-00001", "S3-00001", "S2-00002"}),
    }

    precision, recall, entity_score = entity_f05(gt["S1-00001"], pred["S1-00001"])
    macro = macro_f05(gt, pred)

    print("=" * 60)
    print("PS WORKED EXAMPLE -- Phase 0 gate check")
    print("=" * 60)
    print(f"  Entity:    S1-00001")
    print(f"  GT:        {sorted(gt['S1-00001'])}")
    print(f"  Pred:      {sorted(pred['S1-00001'])}")
    print(f"  Precision: {precision:.6f}  (expected ~0.6667)")
    print(f"  Recall:    {recall:.6f}  (expected 1.0000)")
    print(f"  F0.5:      {entity_score:.6f}  (expected ~0.7143)")
    print(f"  Macro F0.5 (1 entity): {macro:.6f}  (expected ~0.7143)")

    EXPECTED_F05 = 0.7143
    TOLERANCE = 5e-4
    assert abs(macro - EXPECTED_F05) < TOLERANCE, (
        f"GATE FAILED: macro_f05={macro:.6f} is not within {TOLERANCE} of {EXPECTED_F05}. "
        "Fix scorer before advancing."
    )
    print()
    print("GATE PASSED: macro_f05 reproduces PS worked example within tolerance.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_self_test()
