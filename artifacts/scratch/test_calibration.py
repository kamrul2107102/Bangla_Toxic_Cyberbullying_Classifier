import math
from collections import Counter
import numpy as np
import pandas as pd
from src.config import LABELS, SEED
from src.preprocessing import normalize_text, TOKEN_RE
from src.labels import to_arrays
from src.train import split_indices

BANGLA_STOPWORDS = {
    "এই", "সেই", "একটি", "একটা", "কি", "কী", "কেন", "কোথায়", "কোথায়", "কিভাবে",
    "কখন", "কোন", "কোনো", "যে", "সে", "তিনি", "তারা", "তাদের", "তার", "তাকে",
    "আমরা", "আমাদের", "তোমরা", "তোমাদের", "আপনি", "আপনার", "আপনারা",
    "হবে", "হলে", "হলো", "হচ্ছে", "হয়েছে", "আছে", "ছিল", "থাকবে",
    "যাবে", "যেতে", "থেকে", "দিয়ে", "করে", "করতে", "করা", "এবং", "বা",
    "কিন্তু", "যদি", "তবে", "তাহলে", "আর", "তো", "ও"
}

def custom_preprocess(text: str):
    text = normalize_text(text)
    toks = TOKEN_RE.findall(text)
    return [t for t in toks if t not in BANGLA_STOPWORDS]

def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))

class TestSparseTfidf:
    def __init__(self, max_features=12000, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab = {}
        self.idf = None

    def fit(self, texts):
        df = Counter()
        tf_totals = Counter()
        for text in texts:
            toks = custom_preprocess(text)
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
            toks = custom_preprocess(text)
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

class BalancedLogReg:
    def __init__(self, n_features, lr=0.1, epochs=8, l2=1e-5, seed=SEED):
        self.n_features = n_features
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((len(LABELS), n_features), dtype=np.float32)
        self.bias = np.full(len(LABELS), -1.0, dtype=np.float32)

    def fit(self, X_rows, y, mask):
        n = len(X_rows)
        # Compute class weights per label
        weights_pos = np.ones(len(LABELS), dtype=np.float32)
        weights_neg = np.ones(len(LABELS), dtype=np.float32)
        for j in range(len(LABELS)):
            known = mask[:, j] == 1
            pos = float(np.sum((y[:, j] == 1) & known))
            neg = float(np.sum((y[:, j] == 0) & known))
            if pos > 0 and neg > 0:
                tot = pos + neg
                # Give slightly higher penalty to false positives to protect neutral sentences
                weights_pos[j] = tot / (2.0 * pos)
                weights_neg[j] = 1.2 * tot / (2.0 * neg)

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

# Load data and run fast test
df = pd.read_csv("data/processed/merged_dataset.csv")
train_idx, val_idx, test_idx = split_indices(df)
texts = df["text"].fillna("").astype(str).tolist()
y, mask = to_arrays(df.to_dict("records"))

tr = train_idx[:10000] # fast subset for test
vec = TestSparseTfidf().fit([texts[i] for i in tr])
X_tr = vec.transform([texts[i] for i in tr])

clf = BalancedLogReg(len(vec.vocab), epochs=4)
clf.fit(X_tr, y[tr], mask[tr])

test_sentences = [
    "এই বইটা কোথায় কিনতে পাওয়া যাবে?",
    "আজকের আবহাওয়াটা অনেক সুন্দর এবং চমৎকার।",
    "ভাই আপনার কাজটি সত্যিই অনেক তথ্যবহুল ও দারুণ ছিল!",
    "তুই একটা আস্ত কুত্তার বাচ্চা তোরে জুতা মারমু",
    "তোরে যেখানে পামু মাইরা তক্তা বানায়া ফেলমু"
]

X_test = vec.transform(test_sentences)
preds = clf.predict_proba(X_test)

for s, p in zip(test_sentences, preds):
    print("---")
    print("Sentence:", s)
    res = {LABELS[i]: f"{p[i]*100:.1f}%" for i in range(len(LABELS))}
    print("Predictions:", res)
