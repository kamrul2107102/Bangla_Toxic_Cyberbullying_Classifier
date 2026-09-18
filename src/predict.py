from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from .config import ARTIFACT_DIR, LABELS
from .preprocessing import preprocess


def load_model(model_name: str):
    model = joblib.load(ARTIFACT_DIR / f"{model_name}.joblib")
    # A model trained on GPU should still be usable by a CPU-only Streamlit machine.
    if hasattr(model, "model") and hasattr(model, "device") and model.model is not None:
        try:
            model.device = "cpu"
            model.model.to("cpu")
        except Exception:
            pass
    return model


def predict(model, text: str, threshold=0.5):
    probs = np.asarray(model.predict_proba([text]))[0]
    if isinstance(threshold, dict):
        labels = [LABELS[i] for i, p in enumerate(probs) if p >= threshold.get(LABELS[i], 0.5)]
    else:
        labels = [LABELS[i] for i, p in enumerate(probs) if p >= threshold]
    return {
        "labels": labels,
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
        "neutral": len(labels) == 0,
        "tokens": preprocess(text),
    }
