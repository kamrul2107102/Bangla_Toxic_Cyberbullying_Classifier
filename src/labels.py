from __future__ import annotations

from typing import Iterable
import numpy as np
from .config import LABELS


def blank_labels():
    values = {k: -1 for k in LABELS}
    masks = {f"{k}_mask": 0 for k in LABELS}
    return values | masks


def any_toxic(row: dict) -> int:
    return int(any(row.get(k, 0) == 1 for k in LABELS if k != "toxic"))


def to_arrays(rows: list[dict]):
    y = np.array([[int(r[l]) if r[l] in (0, 1) else 0 for l in LABELS] for r in rows], dtype=np.float32)
    mask = np.array([[int(r[f"{l}_mask"]) for l in LABELS] for r in rows], dtype=np.float32)
    return y, mask


def clean_binary(value) -> int | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "positive", "toxic", "hs", "hate"}:
        return 1
    if s in {"0", "false", "no", "n", "negative", "non-toxic", "non toxic", "nh", "not hate", "neutral"}:
        return 0
    try:
        v = float(s)
        if v in (0.0, 1.0):
            return int(v)
    except ValueError:
        pass
    return None


def parse_multilabel_string(value: str) -> set[str]:
    if value is None:
        return set()
    s = str(value).lower()
    for sep in ["|", ";", ",", "/"]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep)]
            return set(parts)
    return {s.strip()} if s.strip() else set()
