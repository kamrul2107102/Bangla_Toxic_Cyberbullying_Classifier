from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from .config import LABELS, PROCESSED_DIR, ARTIFACT_DIR, SEED
from .labels import to_arrays
from .data_splits import get_or_create_splits
from .evaluation.threshold_tuning import tune_thresholds_on_validation, apply_thresholds_to_test
from .models.tfidf_lr import TfidfLogRegModel
from .models.naive_bayes import MultiLabelMultinomialNB
from .models.word2vec_lr import Word2VecLogRegModel


def count_parameters(model) -> int:
    """Calculate trainable parameter count if applicable."""
    if hasattr(model, "network") and model.network is not None:
        return sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    if hasattr(model, "model") and model.model is not None and hasattr(model.model, "parameters"):
        return sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    if hasattr(model, "classifier") and hasattr(model.classifier, "weights"):
        return int(model.classifier.weights.size + model.classifier.bias.size)
    if hasattr(model, "weights"):
        return int(model.weights.size + getattr(model, "bias", np.array([])).size)
    if hasattr(model, "vocab") and hasattr(model, "log_prob_pos") and model.log_prob_pos is not None:
        return int(model.log_prob_pos.size * 2 + model.log_prior.size)
    return 0


def build_model(model_name: str, epochs: int | None = None):
    if model_name == "tfidf_lr":
        return TfidfLogRegModel(epochs=epochs or 8)
    elif model_name == "naive_bayes":
        return MultiLabelMultinomialNB()
    elif model_name == "word2vec_lr":
        return Word2VecLogRegModel(epochs_w2v=epochs or 2, epochs_lr=20)
    elif model_name == "bilstm":
        from .models.bilstm import BiLSTMModel
        return BiLSTMModel(epochs=epochs or 6)
    elif model_name == "transformer":
        from .models.transformer import ScratchTransformerModel
        return ScratchTransformerModel(epochs=epochs or 6)
    elif model_name == "banglabert":
        from .models.banglabert import BanglaBERTBenchmarkModel
        return BanglaBERTBenchmarkModel(epochs=epochs or 3)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run(model_name: str, df: pd.DataFrame, epochs: int | None = None):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Zero data leakage: get or create frozen train/val/test splits
    train_idx, val_idx, test_idx = get_or_create_splits(df)
    texts = df["text"].fillna("").astype(str).tolist()
    y, mask = to_arrays(df.to_dict("records"))

    tr, va, te = train_idx, val_idx, test_idx
    train_texts = [texts[i] for i in tr]
    val_texts = [texts[i] for i in va]
    test_texts = [texts[i] for i in te]

    model = build_model(model_name, epochs)
    display_name = getattr(model, "name", model_name)
    is_benchmark = getattr(model, "is_external_benchmark", False)

    print(f"\n========================================================")
    print(f"Training: {display_name}")
    if is_benchmark:
        print(">> Note: External Pretrained Benchmark (CSE BUET NLP ELECTRA)")
    print(f"Train samples: {len(tr)} | Val: {len(va)} | Test: {len(te)}")
    print(f"========================================================")

    # 2. Train with timing
    t0 = time.perf_counter()
    model.fit(train_texts, y[tr], mask[tr])
    train_time_sec = time.perf_counter() - t0

    # 3. Predict on validation set
    val_probs = model.predict_proba(val_texts)

    # 4. Strict validation threshold tuning (ZERO test leakage)
    best_thresholds = tune_thresholds_on_validation(y[va], val_probs, mask[va]) if len(va) else {l: 0.5 for l in LABELS}
    val_metrics = apply_thresholds_to_test(y[va], val_probs, mask[va], best_thresholds)

    # 5. Measure test inference latency
    t_inf_start = time.perf_counter()
    test_probs = model.predict_proba(test_texts)
    test_inf_time_total = time.perf_counter() - t_inf_start
    inf_ms_per_sample = (test_inf_time_total / max(len(te), 1)) * 1000.0

    # 6. Evaluate test set strictly with validation-tuned thresholds
    test_metrics = apply_thresholds_to_test(y[te], test_probs, mask[te], best_thresholds)

    # 7. Print summary metrics
    print(f"\n--- Evaluation Results [{display_name}] ---")
    print(f"Train Time: {train_time_sec:.2f}s | Inference: {inf_ms_per_sample:.2f} ms/sample")
    print(f"Validation Micro-F1: {val_metrics['micro_f1']:.4f} | Macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"Test Micro-F1:       {test_metrics['micro_f1']:.4f} | Macro-F1: {test_metrics['macro_f1']:.4f}")
    print("Validation-tuned thresholds:", best_thresholds)
    for l in LABELS:
        res = test_metrics["per_label"][l]
        print(f"  {l:12s} - F1: {res['f1']:.4f}, Prec: {res['precision']:.4f}, Rec: {res['recall']:.4f} (support {res['support']})")

    # 8. Save structured artifacts
    model_dir = ARTIFACT_DIR / model_name
    if hasattr(model, "save"):
        model.save(model_dir)

    joblib_path = ARTIFACT_DIR / f"{model_name}.joblib"
    joblib.dump(model, joblib_path)
    model_size_mb = joblib_path.stat().st_size / (1024 * 1024)

    param_count = count_parameters(model)

    report = {
        "model": model_name,
        "display_name": display_name,
        "is_external_benchmark": is_benchmark,
        "train_samples": int(len(tr)),
        "val_samples": int(len(va)),
        "test_samples": int(len(te)),
        "train_time_sec": round(train_time_sec, 2),
        "inference_ms_per_sample": round(inf_ms_per_sample, 3),
        "trainable_parameters": param_count,
        "model_size_mb": round(model_size_mb, 2),
        "best_thresholds": best_thresholds,
        "validation": val_metrics,
        "test": test_metrics,
        "labels": LABELS,
    }
    metrics_path = ARTIFACT_DIR / f"{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nArtifacts saved to:\n  - {joblib_path}\n  - {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description="Unified Train Pipeline for Bangla Toxic Classifier")
    parser.add_argument("--data", default=str(PROCESSED_DIR / "merged_dataset.csv"))
    parser.add_argument(
        "--model",
        choices=["naive_bayes", "tfidf_lr", "word2vec_lr", "bilstm", "transformer", "banglabert"],
        default="tfidf_lr"
    )
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    run(args.model, df, args.epochs)


if __name__ == "__main__":
    main()
