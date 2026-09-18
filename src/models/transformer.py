from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .bilstm import TokenVocab, SequenceDataset, masked_bce_loss
from ..config import LABELS, MAX_VOCAB, MAX_LEN, SEED


if TORCH_AVAILABLE:
    class PositionalEncoding(nn.Module):
        """Standard sinusoidal positional encoding for sequence tokens."""
        def __init__(self, d_model: int, max_len: int = MAX_LEN):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x is (batch, seq_len, d_model)
            return x + self.pe[:, :x.size(1)]


    class ScratchTransformerNetwork(nn.Module):
        """
        Transformer Encoder multi-label classifier trained strictly from scratch.
        No Hugging Face pretrained weights.
        """
        def __init__(
            self,
            vocab_size: int,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 256,
            dropout: float = 0.2,
            max_len: int = 64,
            num_labels: int = len(LABELS)
        ):
            super().__init__()
            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
            self.dropout = nn.Dropout(dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.fc = nn.Linear(d_model, num_labels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Mask where tokens are <PAD> (id == 0)
            src_key_padding_mask = x.eq(0)  # (batch, seq_len)
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()  # (batch, seq_len, 1)

            # Embedding + Scale + Positional Encoding
            emb = self.embedding(x) * math.sqrt(self.d_model)
            emb = self.pos_encoder(emb)
            emb = self.dropout(emb)

            # Transformer Encoder Self-Attention
            encoded = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)

            # Masked average pooling across valid tokens
            sum_pooled = (encoded * valid_mask).sum(dim=1)
            lengths = valid_mask.sum(dim=1).clamp_min(1.0)
            pooled = sum_pooled / lengths

            logits = self.fc(self.dropout(pooled))
            return logits
else:
    class PositionalEncoding:
        pass
    class ScratchTransformerNetwork:
        pass


class ScratchTransformerModel:
    """
    Self-contained Learned-from-Scratch Transformer Multi-Label Classifier.
    - .fit(texts, y, mask)
    - .predict_proba(texts)
    - .save(dir_or_file) / .load(dir_or_file)
    """
    name = "Transformer Encoder (from scratch)"

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        max_len: int = 64,
        batch_size: int = 32,
        lr: float = 5e-4,
        epochs: int = 6,
        seed: int = SEED
    ):
        self.config = {
            "model_type": "transformer",
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "max_len": max_len,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "seed": seed,
            "labels": LABELS,
        }
        self.vocab = TokenVocab(max_vocab=MAX_VOCAB)
        self.network: Optional[ScratchTransformerNetwork] = None
        self.training_history: List[Dict[str, float]] = []
        self.best_thresholds: Dict[str, float] = {lbl: 0.5 for lbl in LABELS}

        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.manual_seed(seed)
        else:
            self.device = "cpu"

    def fit(self, texts: List[str], y: np.ndarray, mask: np.ndarray) -> "ScratchTransformerModel":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to train the Transformer. Please install torch.")

        # 1. Fit vocabulary solely on the training partition
        self.vocab.fit(texts)
        vocab_size = len(self.vocab.itos)

        # 2. Instantiate Transformer architecture
        self.network = ScratchTransformerNetwork(
            vocab_size=vocab_size,
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["dropout"],
            max_len=self.config["max_len"],
            num_labels=len(LABELS)
        ).to(self.device)

        # 3. DataLoader
        dataset = SequenceDataset(texts, y, mask, self.vocab, max_len=self.config["max_len"])
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.config["lr"], weight_decay=1e-4)

        # 4. Training loop
        self.network.train()
        for ep in range(self.config["epochs"]):
            total_loss = 0.0
            for Xb, yb, mb in loader:
                Xb, yb, mb = Xb.to(self.device), yb.to(self.device), mb.to(self.device)
                optimizer.zero_grad()
                logits = self.network(Xb)
                loss = masked_bce_loss(logits, yb, mb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += float(loss.item())

            avg_loss = total_loss / max(len(loader), 1)
            self.training_history.append({"epoch": ep + 1, "train_loss": avg_loss})
            print(f"  [Transformer] Epoch {ep + 1}/{self.config['epochs']} - Loss: {avg_loss:.4f}")

        return self

    def predict_proba(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not TORCH_AVAILABLE or self.network is None:
            raise RuntimeError("Transformer model is not loaded or PyTorch is not available.")

        self.network.eval()
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = [self.vocab.encode(t, max_len=self.config["max_len"]) for t in batch_texts]
                X_tensor = torch.tensor(encoded, dtype=torch.long, device=self.device)
                logits = self.network(X_tensor)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs) if all_probs else np.zeros((0, len(LABELS)), dtype=np.float32)

    def save(self, save_dir: Path | str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.network is not None:
            torch.save(self.network.state_dict(), save_dir / "model.pt")

        (save_dir / "vocab.json").write_text(json.dumps(self.vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (save_dir / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        (save_dir / "history.json").write_text(json.dumps(self.training_history, indent=2), encoding="utf-8")
        (save_dir / "thresholds.json").write_text(json.dumps(self.best_thresholds, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, save_dir: Path | str) -> "ScratchTransformerModel":
        save_dir = Path(save_dir)
        config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
        model = cls(
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            max_len=config["max_len"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            epochs=config["epochs"],
            seed=config.get("seed", SEED)
        )
        vocab_dict = json.loads((save_dir / "vocab.json").read_text(encoding="utf-8"))
        model.vocab = TokenVocab.from_dict(vocab_dict)

        if (save_dir / "thresholds.json").exists():
            model.best_thresholds = json.loads((save_dir / "thresholds.json").read_text(encoding="utf-8"))
        if (save_dir / "history.json").exists():
            model.training_history = json.loads((save_dir / "history.json").read_text(encoding="utf-8"))

        if TORCH_AVAILABLE and (save_dir / "model.pt").exists():
            model.network = ScratchTransformerNetwork(
                vocab_size=len(model.vocab.itos),
                d_model=config["d_model"],
                nhead=config["nhead"],
                num_layers=config["num_layers"],
                dim_feedforward=config["dim_feedforward"],
                dropout=config["dropout"],
                max_len=config["max_len"],
                num_labels=len(LABELS)
            )
            state_dict = torch.load(save_dir / "model.pt", map_location=model.device)
            model.network.load_state_dict(state_dict)
            model.network.to(model.device)
            model.network.eval()

        return model
