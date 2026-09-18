"""
External Pretrained Benchmark: BanglaBERT
Checkpoint: csebuetnlp/banglabert (Bengali ELECTRA Discriminator)

IMPORTANT ACADEMIC CONSTRAINT:
This model uses a pretrained Bengali language model.
It is NOT part of the proposal-compliant from-scratch model family.
It serves exclusively as an external reference benchmark to contextualize
how custom classical and sequence models perform against a large pretrained LM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    BENCHMARK_DEPS_AVAILABLE = True
except ImportError:
    BENCHMARK_DEPS_AVAILABLE = False

from ..config import LABELS, MAX_LEN, SEED


MODEL_CHECKPOINT = "csebuetnlp/banglabert"


class BanglaBERTDataset:
    def __init__(self, texts: List[str], y: np.ndarray, mask: np.ndarray, tokenizer, max_len: int = MAX_LEN):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.y[idx]
        item["mask"] = self.mask[idx]
        return item


class BanglaBERTBenchmarkModel:
    """
    External Pretrained Benchmark wrapper for BanglaBERT (ELECTRA).
    Provides:
    - frozen-base training mode (fast head training for CPU / 8GB RAM)
    - full fine-tuning mode (for Google Colab GPU)
    - standard .predict_proba(texts) interface
    """
    name = "BanglaBERT (External Pretrained Benchmark)"
    is_external_benchmark = True

    def __init__(
        self,
        checkpoint: str = MODEL_CHECKPOINT,
        max_len: int = 64,
        batch_size: int = 16,
        lr: float = 2e-5,
        epochs: int = 3,
        freeze_base: bool = True,
        seed: int = SEED
    ):
        self.config = {
            "model_type": "banglabert_benchmark",
            "checkpoint": checkpoint,
            "max_len": max_len,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "freeze_base": freeze_base,
            "seed": seed,
            "labels": LABELS,
            "is_external_benchmark": True,
        }
        self.tokenizer = None
        self.model = None
        self.best_thresholds: Dict[str, float] = {lbl: 0.5 for lbl in LABELS}
        self.training_history: List[Dict[str, float]] = []

        if BENCHMARK_DEPS_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.manual_seed(seed)
        else:
            self.device = "cpu"

    def _init_components(self):
        if not BENCHMARK_DEPS_AVAILABLE:
            raise RuntimeError(
                "Hugging Face transformers and PyTorch are required for BanglaBERT benchmark. "
                "Install using: pip install -r requirements-benchmark.txt"
            )
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["checkpoint"])
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config["checkpoint"],
                num_labels=len(LABELS),
                problem_type="multi_label_classification",
                ignore_mismatched_sizes=True
            ).to(self.device)

            if self.config["freeze_base"]:
                # Freeze all ELECTRA encoder layers; only train classification head
                for name, param in self.model.named_parameters():
                    if "classifier" not in name:
                        param.requires_grad = False

    def fit(self, texts: List[str], y: np.ndarray, mask: np.ndarray) -> "BanglaBERTBenchmarkModel":
        self._init_components()

        dataset = BanglaBERTDataset(texts, y, mask, self.tokenizer, max_len=self.config["max_len"])
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.config["lr"], weight_decay=0.01)

        self.model.train()
        for ep in range(self.config["epochs"]):
            total_loss = 0.0
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                label_mask = batch["mask"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Masked binary cross entropy
                loss_all = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                loss = (loss_all * label_mask).sum() / label_mask.sum().clamp_min(1.0)

                loss.backward()
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())

            avg_loss = total_loss / max(len(loader), 1)
            self.training_history.append({"epoch": ep + 1, "train_loss": avg_loss})
            print(f"  [BanglaBERT] Epoch {ep + 1}/{self.config['epochs']} - Loss: {avg_loss:.4f}")

        return self

    def predict_proba(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not BENCHMARK_DEPS_AVAILABLE or self.model is None or self.tokenizer is None:
            raise RuntimeError("BanglaBERT model is not initialized.")

        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config["max_len"],
                    return_tensors="pt"
                ).to(self.device)

                logits = self.model(**inputs).logits
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs) if all_probs else np.zeros((0, len(LABELS)), dtype=np.float32)

    def save(self, save_dir: Path | str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

        # Save training metadata to train_config.json without overwriting Hugging Face's config.json
        (save_dir / "train_config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        (save_dir / "history.json").write_text(json.dumps(self.training_history, indent=2), encoding="utf-8")
        (save_dir / "thresholds.json").write_text(json.dumps(self.best_thresholds, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, save_dir: Path | str) -> "BanglaBERTBenchmarkModel":
        save_dir = Path(save_dir)
        cfg_file = save_dir / "train_config.json" if (save_dir / "train_config.json").exists() else save_dir / "config.json"
        config = {}
        if cfg_file.exists():
            try:
                config = json.loads(cfg_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        model = cls(
            checkpoint=config.get("checkpoint", MODEL_CHECKPOINT),
            max_len=config.get("max_len", MAX_LEN),
            batch_size=config.get("batch_size", 16),
            lr=config.get("lr", 2e-5),
            epochs=config.get("epochs", 3),
            freeze_base=config.get("freeze_base", True),
            seed=config.get("seed", SEED)
        )
        if (save_dir / "thresholds.json").exists():
            model.best_thresholds = json.loads((save_dir / "thresholds.json").read_text(encoding="utf-8"))
        if (save_dir / "history.json").exists():
            model.training_history = json.loads((save_dir / "history.json").read_text(encoding="utf-8"))

        if BENCHMARK_DEPS_AVAILABLE:
            model.tokenizer = AutoTokenizer.from_pretrained(save_dir)
            try:
                model.model = AutoModelForSequenceClassification.from_pretrained(save_dir)
            except Exception:
                # Fallback: if save_dir/config.json has non-HF keys
                hf_config = AutoConfig.from_pretrained(
                    model.config.get("checkpoint", MODEL_CHECKPOINT),
                    num_labels=len(LABELS),
                    problem_type="multi_label_classification"
                )
                model.model = AutoModelForSequenceClassification.from_pretrained(save_dir, config=hf_config)

            model.model.to(model.device)
            model.model.eval()

        return model
