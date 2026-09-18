from __future__ import annotations

from typing import Dict, Any
import numpy as np

from ..config import LABELS
from ..metrics import metrics_with_thresholds


def _calc_prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def tune_thresholds_on_validation(
    y_val: np.ndarray,
    val_probs: np.ndarray,
    val_mask: np.ndarray,
    grid: np.ndarray | None = None
) -> Dict[str, float]:
    """
    Search and select detection thresholds per label using ONLY the validation dataset.
    This guarantees zero data leakage into test evaluation.
    
    Args:
        y_val: binary ground-truth labels for validation set (N, num_labels)
        val_probs: predicted probabilities for validation set (N, num_labels)
        val_mask: binary indicator of known labels for validation set (N, num_labels)
        grid: candidate threshold values to scan
    
    Returns:
        Dictionary mapping each canonical label to its best validation threshold.
    """
    if grid is None:
        grid = np.arange(0.15, 0.85, 0.05)

    y_val = np.asarray(y_val)
    val_probs = np.asarray(val_probs)
    val_mask = np.asarray(val_mask)

    best_thresholds: Dict[str, float] = {}

    for j, label in enumerate(LABELS):
        known = val_mask[:, j] == 1
        if not known.any():
            best_thresholds[label] = 0.5
            continue

        yt = y_val[known, j].astype(int)
        best_t = 0.5
        best_f = -1.0

        for t in grid:
            yp = (val_probs[known, j] >= t).astype(int)
            tp = int(((yt == 1) & (yp == 1)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            _, _, f = _calc_prf(tp, fp, fn)

            if f > best_f:
                best_f = float(f)
                best_t = float(t)

        best_thresholds[label] = round(best_t, 4)

    return best_thresholds


def apply_thresholds_to_test(
    y_test: np.ndarray,
    test_probs: np.ndarray,
    test_mask: np.ndarray,
    thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """
    Evaluate test predictions using FROZEN thresholds tuned on validation data.
    """
    return metrics_with_thresholds(y_test, test_probs, test_mask, thresholds)
