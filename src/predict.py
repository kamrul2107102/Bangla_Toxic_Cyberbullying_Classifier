from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
from .config import ARTIFACT_DIR, LABELS
from .preprocessing import preprocess


def load_model(model_name: str):
    """
    Load any classical, neural sequence, or external benchmark model.
    Supports both single-file joblib and structured artifact directories.
    """
    joblib_path = ARTIFACT_DIR / f"{model_name}.joblib"
    model_dir = ARTIFACT_DIR / model_name

    # Neural and benchmark models are best loaded from their structured artifact directory
    if model_dir.is_dir() and model_name in ["banglabert", "bilstm", "transformer"]:
        if model_name == "bilstm":
            from .models.bilstm import BiLSTMModel
            model = BiLSTMModel.load(model_dir)
        elif model_name == "transformer":
            from .models.transformer import ScratchTransformerModel
            model = ScratchTransformerModel.load(model_dir)
        elif model_name == "banglabert":
            from .models.banglabert import BanglaBERTBenchmarkModel
            model = BanglaBERTBenchmarkModel.load(model_dir)
        else:
            raise FileNotFoundError(f"Unknown model directory format for {model_name}")
    elif joblib_path.exists():
        try:
            model = joblib.load(joblib_path)
        except Exception as e:
            if model_dir.is_dir():
                if model_name == "banglabert":
                    from .models.banglabert import BanglaBERTBenchmarkModel
                    model = BanglaBERTBenchmarkModel.load(model_dir)
                elif model_name == "bilstm":
                    from .models.bilstm import BiLSTMModel
                    model = BiLSTMModel.load(model_dir)
                elif model_name == "transformer":
                    from .models.transformer import ScratchTransformerModel
                    model = ScratchTransformerModel.load(model_dir)
                else:
                    raise e
            else:
                raise e
    elif model_dir.is_dir():
        if model_name == "bilstm":
            from .models.bilstm import BiLSTMModel
            model = BiLSTMModel.load(model_dir)
        elif model_name == "transformer":
            from .models.transformer import ScratchTransformerModel
            model = ScratchTransformerModel.load(model_dir)
        elif model_name == "banglabert":
            from .models.banglabert import BanglaBERTBenchmarkModel
            model = BanglaBERTBenchmarkModel.load(model_dir)
        else:
            raise FileNotFoundError(f"Unknown model directory format for {model_name}")
    else:
        raise FileNotFoundError(f"No trained artifact found for model: '{model_name}' in {ARTIFACT_DIR}")

    # Fallback to CPU for PyTorch models trained on GPU
    for attr in ["model", "network"]:
        net = getattr(model, attr, None)
        if net is not None and hasattr(net, "to"):
            try:
                model.device = "cpu"
                net.to("cpu")
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
