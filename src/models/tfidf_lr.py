from __future__ import annotations
import math
from collections import Counter
import numpy as np
from ..preprocessing import preprocess
from ..config import LABELS, MAX_VOCAB, SEED


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


class SparseTfidf:
    def __init__(self, max_features=MAX_VOCAB, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab = {}
        self.idf = None

    def fit(self, texts):
        df = Counter()
        tf_totals = Counter()
        for text in texts:
            toks = preprocess(text)
            counts = Counter(toks)
            for tok in counts:
                df[tok] += 1
            tf_totals.update(counts)
        kept = [w for w, c in df.items() if c >= self.min_df]
        kept.sort(key=lambda w: (-tf_totals[w], w))
        kept = kept[: self.max_features]
        self.vocab = {w: i for i, w in enumerate(kept)}
        n = len(texts)
        self.idf = np.ones(len(kept), dtype=np.float32)
        for w, i in self.vocab.items():
            self.idf[i] = math.log((n + 1) / (df[w] + 1)) + 1.0
        return self

    def transform(self, texts):
        rows = []
        for text in texts:
            toks = preprocess(text)
            counts = Counter(toks)
            total = max(sum(counts.values()), 1)
            feats = {}
            for tok, c in counts.items():
                if tok in self.vocab:
                    i = self.vocab[tok]
                    feats[i] = (c / total) * float(self.idf[i])
            norm = math.sqrt(sum(v * v for v in feats.values()))
            if norm > 0:
                feats = {i: v / norm for i, v in feats.items()}
            rows.append(feats)
        return rows

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)


class MultiLabelLogReg:
    def __init__(self, n_features, lr=0.1, epochs=8, l2=1e-5, seed=SEED):
        self.n_features = n_features
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((len(LABELS), n_features), dtype=np.float32)
        # Initialize bias to negative log-odds so default baseline without evidence is non-toxic
        self.bias = np.full(len(LABELS), -1.2, dtype=np.float32)

    def fit(self, X_rows, y, mask):
        n = len(X_rows)
        # Compute class balance weights per label to avoid scraping selection bias
        weights_pos = np.ones(len(LABELS), dtype=np.float32)
        weights_neg = np.ones(len(LABELS), dtype=np.float32)
        for j in range(len(LABELS)):
            known = mask[:, j] == 1
            pos = float(np.sum((y[:, j] == 1) & known))
            neg = float(np.sum((y[:, j] == 0) & known))
            if pos > 0 and neg > 0:
                tot = pos + neg
                weights_pos[j] = tot / (2.0 * pos)
                weights_neg[j] = 1.25 * tot / (2.0 * neg)

        order = np.arange(n)
        for epoch in range(self.epochs):
            self.rng.shuffle(order)
            eta = self.lr / (1.0 + 0.2 * epoch)
            for ii in order:
                feats = X_rows[ii]
                if not feats:
                    continue
                idx = np.fromiter(feats.keys(), dtype=np.int64)
                val = np.fromiter(feats.values(), dtype=np.float32)
                for j in range(len(LABELS)):
                    if mask[ii, j] == 0:
                        continue
                    z = float(np.dot(self.weights[j, idx], val) + self.bias[j])
                    p = float(sigmoid(z))
                    w = weights_pos[j] if y[ii, j] == 1 else weights_neg[j]
                    err = (p - float(y[ii, j])) * w
                    self.weights[j, idx] -= eta * (err * val + self.l2 * self.weights[j, idx])
                    self.bias[j] -= eta * err
            print(f"  epoch {epoch+1}/{self.epochs} complete")
        return self

    def predict_proba(self, X_rows):
        probs = np.zeros((len(X_rows), len(LABELS)), dtype=np.float32)
        for i, feats in enumerate(X_rows):
            if not feats:
                probs[i] = sigmoid(self.bias)
                continue
            idx = np.fromiter(feats.keys(), dtype=np.int64)
            val = np.fromiter(feats.values(), dtype=np.float32)
            probs[i] = sigmoid(self.weights[:, idx] @ val + self.bias)
        return probs


class TfidfLogRegModel:
    name = "TF-IDF + Logistic Regression (from scratch)"

    def __init__(self, max_features=MAX_VOCAB, min_df=2, lr=0.1, epochs=8):
        self.vectorizer = SparseTfidf(max_features=max_features, min_df=min_df)
        self.lr_params = dict(lr=lr, epochs=epochs)
        self.classifier = None

    def fit(self, texts, y, mask):
        X = self.vectorizer.fit_transform(texts)
        self.classifier = MultiLabelLogReg(len(self.vectorizer.vocab), **self.lr_params)
        self.classifier.fit(X, y, mask)
        return self

    def predict_proba(self, texts):
        return self.classifier.predict_proba(self.vectorizer.transform(texts))
