from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..preprocessing import preprocess
from ..config import LABELS, MAX_VOCAB, MAX_LEN, SEED


class TokenVocab:
    def __init__(self, max_vocab=MAX_VOCAB, min_freq=2):
        self.max_vocab = max_vocab; self.min_freq = min_freq
        self.stoi = {"<PAD>": 0, "<UNK>": 1}; self.itos = ["<PAD>", "<UNK>"]

    def fit(self, texts):
        from collections import Counter
        c = Counter(tok for t in texts for tok in preprocess(t))
        words = [w for w, n in c.most_common() if n >= self.min_freq][: self.max_vocab - 2]
        for w in words:
            self.stoi[w] = len(self.itos); self.itos.append(w)
        return self

    def encode(self, text, max_len=MAX_LEN):
        ids = [self.stoi.get(w, 1) for w in preprocess(text)][:max_len]
        return ids + [0] * (max_len - len(ids))


class TextDataset(Dataset):
    def __init__(self, texts, y, mask, vocab):
        self.X = torch.tensor([vocab.encode(t) for t in texts], dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.mask[idx]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=2, num_labels=len(LABELS)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.2 if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x):
        emb = self.embedding(x)
        _, (h, _) = self.lstm(emb)
        h_fwd = h[-2]; h_bwd = h[-1]
        return self.fc(torch.cat([h_fwd, h_bwd], dim=1))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, num_labels=len(LABELS)):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True, dropout=0.2)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        mask = x.eq(0)
        z = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        z = self.pos(z)
        z = self.encoder(z, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1).float()
        pooled = (z * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        return self.fc(pooled)


def masked_bce(logits, y, mask):
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


class TorchSequenceModel:
    def __init__(self, kind="bilstm", epochs=5, batch_size=64, lr=1e-3, seed=SEED):
        self.kind = kind; self.epochs=epochs; self.batch_size=batch_size; self.lr=lr
        self.vocab = TokenVocab(); self.model = None; self.device = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(seed); np.random.seed(seed)

    def _build(self):
        if self.kind == "bilstm":
            self.model = BiLSTMClassifier(len(self.vocab.itos))
        else:
            self.model = TransformerClassifier(len(self.vocab.itos))
        self.model.to(self.device)

    def fit(self, texts, y, mask):
        self.vocab.fit(texts); self._build()
        ds = TextDataset(texts, y, mask, self.vocab)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for ep in range(self.epochs):
            total = 0.0
            for Xb, yb, mb in dl:
                Xb, yb, mb = Xb.to(self.device), yb.to(self.device), mb.to(self.device)
                opt.zero_grad(); logits = self.model(Xb); loss = masked_bce(logits, yb, mb); loss.backward(); opt.step(); total += float(loss.item())
            print(f"  {self.kind} epoch {ep+1}/{self.epochs} loss={total/max(len(dl),1):.4f}")
        return self

    def predict_proba(self, texts):
        self.model.eval()
        X = torch.tensor([self.vocab.encode(t) for t in texts], dtype=torch.long).to(self.device)
        with torch.no_grad():
            p = torch.sigmoid(self.model(X)).cpu().numpy()
        return p
