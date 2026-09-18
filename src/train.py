from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from .config import LABELS, PROCESSED_DIR, ARTIFACT_DIR, SEED
from .labels import to_arrays
from .metrics import multilabel_metrics, tune_thresholds, metrics_with_thresholds
from .models.tfidf_lr import TfidfLogRegModel
from .models.naive_bayes import MultiLabelMultinomialNB
from .models.word2vec_lr import Word2VecLogRegModel


def split_indices(df, seed=SEED):
    rng = np.random.default_rng(seed)
    # Stratify roughly by whether toxicity is known and positive.
    key = []
    for _, r in df.iterrows():
        known = int(r.get("toxic_mask", 0))
        val = int(r.get("toxic", 0)) if known else -1
        key.append(f"{known}_{val}")
    key = np.array(key)
    train=[]; val=[]; test=[]
    for k in np.unique(key):
        ids = np.where(key == k)[0]; rng.shuffle(ids)
        n=len(ids); a=int(0.8*n); b=int(0.9*n)
        train.extend(ids[:a]); val.extend(ids[a:b]); test.extend(ids[b:])
    return np.array(train, dtype=np.int64), np.array(val, dtype=np.int64), np.array(test, dtype=np.int64)


def run(model_name, df, epochs=None):
    train_idx, val_idx, test_idx = split_indices(df)
    texts = df["text"].fillna("").astype(str).tolist()
    y, mask = to_arrays(df.to_dict("records"))
    tr = train_idx; va = val_idx; te = test_idx
    if model_name == "tfidf_lr":
        model = TfidfLogRegModel(epochs=epochs or 8)
    elif model_name == "naive_bayes":
        model = MultiLabelMultinomialNB()
    elif model_name == "word2vec_lr":
        model = Word2VecLogRegModel(epochs_w2v=epochs or 2, epochs_lr=20)
    elif model_name in {"bilstm", "transformer"}:
        from .models.deep_models import TorchSequenceModel
        model = TorchSequenceModel(kind=model_name, epochs=epochs or 5)
    else:
        raise ValueError(model_name)
    print(f"Training: {getattr(model, 'name', model_name)}")
    model.fit([texts[i] for i in tr], y[tr], mask[tr])
    val_probs = model.predict_proba([texts[i] for i in va])
    test_probs = model.predict_proba([texts[i] for i in te])
    best_thresholds = tune_thresholds(y[va], val_probs, mask[va]) if len(va) else {label: 0.5 for label in LABELS}
    val_metrics = metrics_with_thresholds(y[va], val_probs, mask[va], best_thresholds)
    test_metrics = metrics_with_thresholds(y[te], test_probs, mask[te], best_thresholds)
    print("Validation micro-F1:", round(val_metrics["micro_f1"], 4))
    print("Test micro-F1:", round(test_metrics["micro_f1"], 4))
    print("Test macro-F1:", round(test_metrics["macro_f1"], 4))
    print("Validation-tuned thresholds:", best_thresholds)
    for l in LABELS:
        print(l, round(test_metrics["per_label"][l]["f1"], 4), "support", test_metrics["per_label"][l]["support"])
    ARTIFACT_DIR.mkdir(exist_ok=True)
    path = ARTIFACT_DIR / f"{model_name}.joblib"
    joblib.dump(model, path)
    report = {"model": model_name, "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)), "validation": val_metrics, "test": test_metrics, "labels": LABELS, "best_thresholds": best_thresholds}
    (ARTIFACT_DIR / f"{model_name}_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved", path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROCESSED_DIR / "merged_dataset.csv"))
    parser.add_argument("--model", choices=["tfidf_lr", "naive_bayes", "word2vec_lr", "bilstm", "transformer"], default="tfidf_lr")
    parser.add_argument("--epochs", type=int, default=None)
    args=parser.parse_args()
    df=pd.read_csv(args.data)
    run(args.model, df, args.epochs)

if __name__ == "__main__":
    main()
