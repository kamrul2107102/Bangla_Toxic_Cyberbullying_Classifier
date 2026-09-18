from __future__ import annotations
from collections import Counter
import numpy as np
from ..preprocessing import preprocess
from ..config import LABELS, MAX_W2V_VOCAB, EMBED_DIM, SEED
from .tfidf_lr import MultiLabelLogReg


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


class SkipGramNegativeSampling:
    """Small from-scratch Word2Vec implementation; negative sampling scales better than full softmax."""
    def __init__(self, dim=EMBED_DIM, max_vocab=MAX_W2V_VOCAB, window=2, negatives=5, epochs=2, lr=0.025, min_count=2, seed=SEED):
        self.dim = dim; self.max_vocab = max_vocab; self.window = window; self.negatives = negatives
        self.epochs = epochs; self.lr = lr; self.min_count = min_count; self.rng = np.random.default_rng(seed)
        self.word2id = {}; self.id2word = {}; self.W_in = None; self.W_out = None

    def fit(self, texts):
        token_lists = [preprocess(t) for t in texts]
        counts = Counter(tok for doc in token_lists for tok in doc)
        words = [w for w, c in counts.most_common() if c >= self.min_count][: self.max_vocab]
        self.word2id = {w: i for i, w in enumerate(words)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        V = len(self.word2id)
        if V < 2:
            raise ValueError("Word2Vec vocabulary is too small. Lower min_count or add more data.")
        self.W_in = self.rng.normal(0, 0.1, size=(V, self.dim)).astype(np.float32)
        self.W_out = self.rng.normal(0, 0.1, size=(V, self.dim)).astype(np.float32)
        ids_docs = [[self.word2id[w] for w in doc if w in self.word2id] for doc in token_lists]
        unigram = np.array([max(counts[self.id2word[i]], 1) ** 0.75 for i in range(V)], dtype=np.float64)
        unigram /= unigram.sum()
        for epoch in range(self.epochs):
            order = np.arange(len(ids_docs)); self.rng.shuffle(order)
            lr = self.lr * (1 - epoch / max(self.epochs, 1) * 0.5)
            steps = 0
            for di in order:
                seq = ids_docs[di]
                for pos, center in enumerate(seq):
                    lo, hi = max(0, pos-self.window), min(len(seq), pos+self.window+1)
                    for ctx_pos in range(lo, hi):
                        if ctx_pos == pos: continue
                        context = seq[ctx_pos]
                        # Positive + negative updates.
                        samples = [(context, 1)]
                        negs = self.rng.choice(V, size=self.negatives, p=unigram)
                        samples.extend((int(n), 0) for n in negs)
                        vin = self.W_in[center].copy()
                        for target, label in samples:
                            score = float(np.dot(vin, self.W_out[target]))
                            p = float(sigmoid(score))
                            g = (label - p) * lr
                            self.W_out[target] += g * vin
                            self.W_in[center] += g * self.W_out[target]
                        steps += 1
            print(f"  Word2Vec epoch {epoch+1}/{self.epochs}, pairs={steps}")
        return self

    def doc_embeddings(self, texts):
        X = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            ids = [self.word2id[w] for w in preprocess(text) if w in self.word2id]
            if ids:
                X[i] = self.W_in[ids].mean(axis=0)
        return X


class Word2VecLogRegModel:
    name = "Word2Vec (from scratch) + Logistic Regression (from scratch)"

    def __init__(self, dim=EMBED_DIM, max_vocab=MAX_W2V_VOCAB, epochs_w2v=2, epochs_lr=20, min_count=2):
        self.w2v = SkipGramNegativeSampling(dim=dim, max_vocab=max_vocab, epochs=epochs_w2v, min_count=min_count)
        self.lr_epochs = epochs_lr
        self.classifier = None

    def fit(self, texts, y, mask):
        self.w2v.fit(texts)
        X = self.w2v.doc_embeddings(texts)
        self.classifier = MultiLabelLogReg(X.shape[1], lr=0.05, epochs=self.lr_epochs)
        # Dense rows need dictionary wrapper.
        X_rows = []
        for row in X:
            X_rows.append({i: float(v) for i, v in enumerate(row) if abs(float(v)) > 1e-8})
        self.classifier.fit(X_rows, y, mask)
        return self

    def predict_proba(self, texts):
        X = self.w2v.doc_embeddings(texts)
        rows = [{i: float(v) for i, v in enumerate(row) if abs(float(v)) > 1e-8} for row in X]
        return self.classifier.predict_proba(rows)
