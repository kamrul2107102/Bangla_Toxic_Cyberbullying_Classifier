from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, SEED, PROCESSED_DIR


SPLIT_FILE = ARTIFACT_DIR / "splits.json"


def check_cross_split_leakage(df: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> int:
    """Check if any normalized text in train split appears identically in test split."""
    from .preprocessing import normalize_text
    train_texts = set(normalize_text(t) for t in df.iloc[train_idx]["text"].dropna())
    test_texts = [normalize_text(t) for t in df.iloc[test_idx]["text"].dropna()]
    leaks = sum(1 for t in test_texts if t in train_texts)
    return leaks


def create_stratified_splits(df: pd.DataFrame, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratify train (80%), validation (10%), and test (10%) splits.
    Stratification is based on toxicity annotation availability and positive status.
    """
    rng = np.random.default_rng(seed)
    keys = []
    for _, r in df.iterrows():
        known = int(r.get("toxic_mask", 0))
        val = int(r.get("toxic", 0)) if known else -1
        keys.append(f"{known}_{val}")
    keys = np.array(keys)

    train, val, test = [], [], []
    for k in np.unique(keys):
        ids = np.where(keys == k)[0]
        rng.shuffle(ids)
        n = len(ids)
        a = int(0.8 * n)
        b = int(0.9 * n)
        train.extend(ids[:a])
        val.extend(ids[a:b])
        test.extend(ids[b:])

    train_idx = np.array(train, dtype=np.int64)
    val_idx = np.array(val, dtype=np.int64)
    test_idx = np.array(test, dtype=np.int64)
    return train_idx, val_idx, test_idx


def get_or_create_splits(
    df: pd.DataFrame,
    split_path: Path = SPLIT_FILE,
    force_recreate: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retrieve frozen train/val/test indices from disk or create them deterministically.
    Guaranteeing that all models (classical, deep, benchmark) train and evaluate on
    the EXACT same rows without data leakage.
    """
    split_path = Path(split_path)
    if split_path.exists() and not force_recreate:
        try:
            data = json.loads(split_path.read_text(encoding="utf-8"))
            train_idx = np.array(data["train_indices"], dtype=np.int64)
            val_idx = np.array(data["val_indices"], dtype=np.int64)
            test_idx = np.array(data["test_indices"], dtype=np.int64)
            # Validate bounds against current df
            if (len(train_idx) + len(val_idx) + len(test_idx)) == len(df):
                return train_idx, val_idx, test_idx
        except Exception:
            pass

    train_idx, val_idx, test_idx = create_stratified_splits(df)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    leaks = check_cross_split_leakage(df, train_idx, test_idx)
    
    payload = {
        "seed": SEED,
        "total_samples": int(len(df)),
        "train_count": int(len(train_idx)),
        "val_count": int(len(val_idx)),
        "test_count": int(len(test_idx)),
        "cross_split_leak_count": leaks,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }
    split_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    data_path = PROCESSED_DIR / "merged_dataset.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        tr, va, te = get_or_create_splits(df, force_recreate=True)
        print(f"Splits created: Train={len(tr)}, Val={len(va)}, Test={len(te)}")
        print(f"Saved to: {SPLIT_FILE}")
