import re
import math
from collections import Counter
import numpy as np
import pandas as pd
from src.config import LABELS, SEED
from src.preprocessing import normalize_text, TOKEN_RE, BANGLA_STOPWORDS
from src.labels import to_arrays
from src.train import split_indices

# Colloquial verb normalizer for Bengali
COLLOQUIAL_MAP = {
    r"(\w+)মু\b": r"\1ব",      # পামু -> পাব, ফেলমু -> ফেলব, মারমু -> মারব
    r"(\w+)সি\b": r"\1ছি",     # করসি -> করছি, গেসি -> গেছি
    r"(\w+)সে\b": r"\1ছে",     # করসে -> করেছে
    r"মাইরা\b": "মেরে",
    r"খাইয়া\b": "খেয়ে",
    r"বানায়া\b": "বানিয়ে",
    r"কোপায়া\b": "কোপিয়ে",
    r"আমারে\b": "আমাকে",
    r"তোমারে\b": "তোমাকে",
}

def normalize_colloquial(text: str) -> str:
    for pat, repl in COLLOQUIAL_MAP.items():
        text = re.sub(pat, repl, text)
    return text

def get_ngrams(text: str, n_range=(1, 2)):
    text = normalize_text(text)
    text = normalize_colloquial(text)
    toks = TOKEN_RE.findall(text)
    
    # Filter stopwords for unigrams, but keep order
    unigrams = [t for t in toks if t not in BANGLA_STOPWORDS]
    
    features = list(unigrams)
    if n_range[1] >= 2 and len(toks) >= 2:
        # Generate bigrams from adjacent tokens
        for i in range(len(toks) - 1):
            w1, w2 = toks[i], toks[i+1]
            # Form bigram if at least one is meaningful or it's a known threat phrase
            features.append(f"{w1}_{w2}")
    return features

test_sentence = "তোরে যেখানে পামু মাইরা তক্তা বানায়া ফেলমু, জানে শেষ করে দেব"
print("Extracted ngrams from test sentence:")
print(get_ngrams(test_sentence))
