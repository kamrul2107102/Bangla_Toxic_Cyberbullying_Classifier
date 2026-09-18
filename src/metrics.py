from __future__ import annotations
import numpy as np
from .config import LABELS


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def multilabel_metrics(y_true, y_pred, mask, threshold=0.5):
    y_pred = (np.asarray(y_pred) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    mask = np.asarray(mask).astype(int)
    per = {}
    tps = fps = fns = 0
    for j, label in enumerate(LABELS):
        known = mask[:, j] == 1
        yt, yp = y_true[known, j], y_pred[known, j]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        p, r, f = _prf(tp, fp, fn)
        per[label] = {"precision": p, "recall": r, "f1": f, "support": int(len(yt)), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        tps += tp; fps += fp; fns += fn
    micro_p, micro_r, micro_f1 = _prf(tps, fps, fns)
    valid_labels = [v["f1"] for v in per.values() if v["support"] > 0]
    macro_f1 = float(np.mean(valid_labels)) if valid_labels else 0.0
    fully_known = np.all(mask == 1, axis=1)
    exact_match = float(np.mean(np.all(y_true[fully_known] == y_pred[fully_known], axis=1))) if fully_known.any() else None
    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "exact_match_fully_annotated": exact_match,
        "per_label": per,
    }


def tune_thresholds(y_true, y_prob, mask, grid=None):
    """Pick an independent threshold per label using validation micro-F1 contribution."""
    if grid is None:
        grid = np.arange(0.20, 0.81, 0.05)
    thresholds = {}
    for j, label in enumerate(LABELS):
        known = mask[:, j] == 1
        if not known.any():
            thresholds[label] = 0.5
            continue
        best_t, best_f = 0.5, -1.0
        yt = y_true[known, j].astype(int)
        for t in grid:
            yp = (y_prob[known, j] >= t).astype(int)
            tp = int(((yt == 1) & (yp == 1)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            _, _, f = _prf(tp, fp, fn)
            if f > best_f:
                best_t, best_f = float(t), float(f)
        thresholds[label] = best_t
    return thresholds

def metrics_with_thresholds(y_true, y_prob, mask, thresholds):
    y_prob = np.asarray(y_prob)
    y_true = np.asarray(y_true).astype(int)
    mask = np.asarray(mask).astype(int)
    pred = np.zeros_like(y_prob, dtype=int)
    for j, label in enumerate(LABELS):
        pred[:, j] = (y_prob[:, j] >= float(thresholds.get(label, 0.5))).astype(int)
    per = {}
    tps = fps = fns = 0
    for j, label in enumerate(LABELS):
        known = mask[:, j] == 1
        yt, yp = y_true[known, j], pred[known, j]
        tp = int(((yt == 1) & (yp == 1)).sum()); fp = int(((yt == 0) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum()); tn = int(((yt == 0) & (yp == 0)).sum())
        p, r, f = _prf(tp, fp, fn)
        per[label] = {"precision": p, "recall": r, "f1": f, "support": int(len(yt)), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        tps += tp; fps += fp; fns += fn
    mp, mr, mf = _prf(tps, fps, fns)
    vals = [v["f1"] for v in per.values() if v["support"] > 0]
    macro = float(np.mean(vals)) if vals else 0.0
    fully_known = np.all(mask == 1, axis=1)
    exact = float(np.mean(np.all(y_true[fully_known] == pred[fully_known], axis=1))) if fully_known.any() else None
    return {"micro_precision": mp, "micro_recall": mr, "micro_f1": mf, "macro_f1": macro, "exact_match_fully_annotated": exact, "per_label": per}
