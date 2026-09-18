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
    "কিন্তু", "যদি", "তবে", "তাহলে", "আর", "তো", "ও", "না", "নি", "নাই", "নেই"
}

def custom_preprocess(text: str):
    text = normalize_text(text)
    toks = TOKEN_RE.findall(text)
    return [t for t in toks if t not in BANGLA_STOPWORDS]

class CalibratedNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.vocab = {}
        self.log_prob_pos = None
        self.log_prob_neg = None
        self.log_prior = np.full(len(LABELS), math.log(0.3), dtype=np.float32) # prior probability of toxicity ~ 30%

    def fit(self, texts, y, mask):
        df = Counter()
        token_counts = []
        for text in texts:
            toks = custom_preprocess(text)
            c = Counter(toks)
            token_counts.append(c)
            df.update(c.keys())
        self.vocab = {w: i for i, (w, _) in enumerate(df.most_common())}
        V = max(len(self.vocab), 1)

        self.log_prob_pos = np.zeros((len(LABELS), V), dtype=np.float32)
        self.log_prob_neg = np.zeros((len(LABELS), V), dtype=np.float32)
        for j in range(len(LABELS)):
            known = mask[:, j] == 1
            cp_docs = Counter(); cn_docs = Counter()
            tp = tn = 0
            for i, text in enumerate(texts):
                if not known[i]:
                    continue
                c = token_counts[i]
                if y[i, j] == 1:
                    cp_docs.update(c); tp += sum(c.values())
                else:
                    cn_docs.update(c); tn += sum(c.values())
            denp = tp + self.alpha * V
            denn = tn + self.alpha * V
            for w, idx in self.vocab.items():
                self.log_prob_pos[j, idx] = math.log((cp_docs.get(w, 0) + self.alpha) / denp)
                self.log_prob_neg[j, idx] = math.log((cn_docs.get(w, 0) + self.alpha) / denn)
        return self

    def predict_proba(self, texts):
        out = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
        for i, text in enumerate(texts):
            c = Counter(custom_preprocess(text))
            for j in range(len(LABELS)):
                pos = float(self.log_prior[j])
                neg = float(math.log(1.0 - math.exp(self.log_prior[j])))
                for w, cnt in c.items():
                    idx = self.vocab.get(w)
                    if idx is not None:
                        pos += cnt * float(self.log_prob_pos[j, idx])
                        neg += cnt * float(self.log_prob_neg[j, idx])
                m = max(pos, neg)
                ep, en = math.exp(pos - m), math.exp(neg - m)
                out[i, j] = ep / (ep + en)
        return out

df = pd.read_csv("data/processed/merged_dataset.csv")
train_idx, val_idx, test_idx = split_indices(df)
texts = df["text"].fillna("").astype(str).tolist()
y, mask = to_arrays(df.to_dict("records"))

nb = CalibratedNB(alpha=1.0)
nb.fit([texts[i] for i in train_idx[:12000]], y[train_idx[:12000]], mask[train_idx[:12000]])

test_sentences = [
    "এই বইটা কোথায় কিনতে পাওয়া যাবে?",
    "আজকের আবহাওয়াটা অনেক সুন্দর এবং চমৎকার।",
    "ভাই আপনার কাজটি সত্যিই অনেক তথ্যবহুল ও দারুণ ছিল!",
    "তুই একটা আস্ত কুত্তার বাচ্চা তোরে জুতা মারমু",
    "তোরে যেখানে পামু মাইরা তক্তা বানায়া ফেলমু"
]

preds = nb.predict_proba(test_sentences)
for s, p in zip(test_sentences, preds):
    print("---")
    print("Sentence:", s)
    res = {LABELS[i]: f"{p[i]*100:.1f}%" for i in range(len(LABELS))}
    print("NB Predictions:", res)
