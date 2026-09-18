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

from ..preprocessing import preprocess
from ..config import LABELS, MAX_VOCAB, MAX_LEN, SEED


class TokenVocab:
    """Lightweight vocabulary builder fitted strictly on training data."""
    def __init__(self, max_vocab: int = MAX_VOCAB, min_freq: int = 2):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.itos: List[str] = ["<PAD>", "<UNK>"]

    def fit(self, texts: List[str]) -> "TokenVocab":
        from collections import Counter
        c = Counter(tok for t in texts for tok in preprocess(t))
        words = [w for w, n in c.most_common() if n >= self.min_freq][: self.max_vocab - 2]
        for w in words:
            self.stoi[w] = len(self.itos)
            self.itos.append(w)
        return self

    def encode(self, text: str, max_len: int = MAX_LEN) -> List[int]:
        ids = [self.stoi.get(w, 1) for w in preprocess(text)][:max_len]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids

    def to_dict(self) -> Dict[str, Any]:
        return {"max_vocab": self.max_vocab, "min_freq": self.min_freq, "stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenVocab":
        vocab = cls(max_vocab=data["max_vocab"], min_freq=data["min_freq"])
        vocab.stoi = data["stoi"]
        vocab.itos = data["itos"]
        return vocab


if TORCH_AVAILABLE:
    class BiLSTMNetwork(nn.Module):
        """
        Bidirectional LSTM neural classifier for multi-label text classification.
        Trained entirely from scratch without pretrained embeddings.
        """
        def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 128,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.2,
            num_labels: int = len(LABELS)
        ):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.dropout = nn.Dropout(dropout)
            self.lstm = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            # Bidirectional LSTM has forward + backward outputs (hidden_dim * 2)
            self.fc = nn.Linear(hidden_dim * 2, num_labels)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mask = x.ne(0).unsqueeze(-1).float()  # (batch, seq_len, 1)
            emb = self.dropout(self.embedding(x))  # (batch, seq_len, embed_dim)
            out, (h, _) = self.lstm(emb)  # out: (batch, seq_len, hidden_dim * 2)
            
            # Masked average pooling across valid tokens for robust representation
            sum_pooled = (out * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp_min(1.0)
            pooled = sum_pooled / lengths
            
            logits = self.fc(self.dropout(pooled))
            return logits

    class SequenceDataset(Dataset):
        def __init__(self, texts: List[str], y: np.ndarray, mask: np.ndarray, vocab: TokenVocab, max_len: int = MAX_LEN):
            self.X = torch.tensor([vocab.encode(t, max_len=max_len) for t in texts], dtype=torch.long)
            self.y = torch.tensor(y, dtype=torch.float32)
            self.mask = torch.tensor(mask, dtype=torch.float32)

        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx], self.mask[idx]

    def masked_bce_loss(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Numerically stable binary cross-entropy with logits, respecting missing label mask."""
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
else:
    class BiLSTMNetwork:
        pass
    class SequenceDataset:
        pass
    def masked_bce_loss(*args, **kwargs):
        pass


class BiLSTMModel:
    """
    Self-contained BiLSTM Multi-Label Classifier.
    Fully compatible with the from-scratch project interface:
    - .fit(texts, y, mask)
    - .predict_proba(texts)
    - .save(dir_or_file) / .load(dir_or_file)
    """
    name = "BiLSTM (from scratch)"

    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_len: int = 64,
        batch_size: int = 32,
        lr: float = 1e-3,
        epochs: int = 6,
        seed: int = SEED
    ):
        self.config = {
            "model_type": "bilstm",
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "max_len": max_len,
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "seed": seed,
            "labels": LABELS,
        }
        self.vocab = TokenVocab(max_vocab=MAX_VOCAB)
        self.network: Optional[BiLSTMNetwork] = None
        self.training_history: List[Dict[str, float]] = []
        self.best_thresholds: Dict[str, float] = {lbl: 0.5 for lbl in LABELS}
        
        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            torch.manual_seed(seed)
        else:
            self.device = "cpu"

    def fit(self, texts: List[str], y: np.ndarray, mask: np.ndarray, val_data: tuple | None = None) -> "BiLSTMModel":
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to train BiLSTM. Please install torch.")
        
        # 1. Fit vocabulary strictly on the training partition
        self.vocab.fit(texts)
        vocab_size = len(self.vocab.itos)
        
        # 2. Instantiate BiLSTM neural architecture
        self.network = BiLSTMNetwork(
            vocab_size=vocab_size,
            embed_dim=self.config["embed_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            num_labels=len(LABELS)
        ).to(self.device)

        # 3. Create PyTorch DataLoader
        dataset = SequenceDataset(texts, y, mask, self.vocab, max_len=self.config["max_len"])
        loader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.config["lr"], weight_decay=1e-5)

        # 4. Training loop with loss logging
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
            print(f"  [BiLSTM] Epoch {ep + 1}/{self.config['epochs']} - Loss: {avg_loss:.4f}")

        return self

    def predict_proba(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        if not TORCH_AVAILABLE or self.network is None:
            raise RuntimeError("BiLSTM model is not loaded or PyTorch is not available.")
        
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
        """Save weights, vocabulary, configuration, thresholds, and training history."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.network is not None:
            torch.save(self.network.state_dict(), save_dir / "model.pt")

        (save_dir / "vocab.json").write_text(json.dumps(self.vocab.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        (save_dir / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        (save_dir / "history.json").write_text(json.dumps(self.training_history, indent=2), encoding="utf-8")
        (save_dir / "thresholds.json").write_text(json.dumps(self.best_thresholds, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, save_dir: Path | str) -> "BiLSTMModel":
        save_dir = Path(save_dir)
        config = json.loads((save_dir / "config.json").read_text(encoding="utf-8"))
        model = cls(
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
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
            model.network = BiLSTMNetwork(
                vocab_size=len(model.vocab.itos),
                embed_dim=config["embed_dim"],
                hidden_dim=config["hidden_dim"],
                num_layers=config["num_layers"],
                dropout=config["dropout"],
                num_labels=len(LABELS)
            )
            state_dict = torch.load(save_dir / "model.pt", map_location=model.device)
            model.network.load_state_dict(state_dict)
            model.network.to(model.device)
            model.network.eval()

        return model
