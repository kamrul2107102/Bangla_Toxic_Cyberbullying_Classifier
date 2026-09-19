from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import json

from src.config import ARTIFACT_DIR, PROCESSED_DIR
from src.preprocessing import normalize_text

# High-risk threat and toxic keywords for pre-moderation warning
TOXIC_KEYWORDS = {
    "মেরে", "মারব", "মারমু", "ফেলব", "ফেলমু", "জবাই", "কুত্তা", "জুতা", "খুন",
    "তক্তা", "ফালতু", "শালা", "হারামি", "বেশ্যা", "খানকির", "শুয়োরের",
    "পুতে", "উড়িয়ে", "উড়াইয়া", "কাটা", "মরণ", "লাথি", "ধ্বংস", "খারাপ"
}

def tokenize_for_lm(text: str) -> list[str]:
    """Tokenize while preserving stop words to retain syntactic transitional sequences (Lab 2)."""
    norm = normalize_text(text)
    return [w for w in norm.split() if w.strip()]

def build_ngram_model(tokens: list[str], n: int = 2) -> dict:
    """
    Builds an N-gram probability lookup table using Maximum Likelihood Estimation (MLE).
    Harkens directly back to NLP Lab 2.
    """
    model = defaultdict(Counter)
    
    # 1. Slide window across tokens and count transitions
    for i in range(len(tokens) - n + 1):
        history = tuple(tokens[i : i + n - 1])
        next_word = tokens[i + n - 1]
        model[history][next_word] += 1
        
    # 2. Convert absolute counts to MLE probabilities: P(w_i | w_{i-n+1}..w_{i-1})
    prob_model = defaultdict(dict)
    for history, context in model.items():
        total_count = sum(context.values())
        for next_word, count in context.items():
            prob_model[history][next_word] = count / total_count
            
    return prob_model

def shannons_predictor(history_phrase: str, model: dict, n: int = 2, top_k: int = 5) -> list[tuple[str, float]]:
    """
    Claude Shannon's next-word guessing experiment (NLP Lab 2).
    Extracts the trailing (n-1) tokens and retrieves next-word candidates sorted by probability.
    """
    tokens = tokenize_for_lm(history_phrase)
    if len(tokens) < n - 1:
        return []
        
    history = tuple(tokens[-(n - 1) :])
    if history in model:
        candidates = model[history]
        sorted_preds = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return sorted_preds[:top_k]
    return []

def generate_text(model: dict, seed_phrase: str, max_words: int = 5, n: int = 2) -> str:
    """
    Autoregressively generates text sequences using np.random.choice (NLP Lab 2).
    """
    output_tokens = tokenize_for_lm(seed_phrase)
    if not output_tokens:
        return seed_phrase
        
    for _ in range(max_words):
        if len(output_tokens) < n - 1:
            break
        history = tuple(output_tokens[-(n - 1) :])
        if history in model:
            choices = list(model[history].keys())
            probabilities = list(model[history].values())
            next_word = np.random.choice(choices, p=probabilities)
            output_tokens.append(next_word)
        else:
            break
            
    return " ".join(output_tokens)

def toxic_autocomplete_guard(history_phrase: str, model: dict, n: int = 2, top_k: int = 5) -> dict:
    """
    Real-Time Pre-moderation & Toxic Auto-complete Warning.
    Uses Shannon's predictor to evaluate next words and flags potential abuse before posting.
    """
    predictions = shannons_predictor(history_phrase, model, n=n, top_k=top_k)
    risky_words = []
    
    for word, prob in predictions:
        if word in TOXIC_KEYWORDS:
            risky_words.append((word, prob))
            
    is_warning = len(risky_words) > 0
    warning_msg = None
    if is_warning:
        top_risky, top_prob = risky_words[0]
        warning_msg = f"⚠️ প্রি-মডারেশন সতর্কতা: পরবর্তী সম্ভাব্য শব্দ '{top_risky}' ({top_prob*100:.1f}% সম্ভাবনা) ক্ষতিকর/আক্রমণাত্মক হতে পারে!"
        
    return {
        "predictions": predictions,
        "is_warning": is_warning,
        "warning_message": warning_msg,
        "risky_words": risky_words
    }

def train_or_load_shannon_guard(n: int = 2, max_rows: int = 20000) -> dict:
    """Trains an N-gram model on the toxic comments dataset or loads from disk."""
    cache_file = ARTIFACT_DIR / f"shannon_{n}gram_model.json"
    
    # 1. Load from cache if available
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            # Reconstruct tuple keys for dictionary lookup
            model = defaultdict(dict)
            for k, v in raw_data.items():
                history_tuple = tuple(k.split("|||"))
                model[history_tuple] = v
            return model
        except Exception:
            pass

    # 2. Otherwise train on dataset
    data_path = PROCESSED_DIR / "merged_dataset.csv"
    tokens = []
    if data_path.exists():
        import pandas as pd
        df = pd.read_csv(data_path, nrows=max_rows)
        # Focus especially on comments with toxicity / threats for strong auto-complete detection
        toxic_comments = df["text"].dropna().astype(str).tolist()
        for text in toxic_comments:
            tokens.extend(tokenize_for_lm(text))
            tokens.append("</s>") # Sentence boundary marker from Lab 2
            
    model = build_ngram_model(tokens, n=n)
    
    # Save cache
    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        serializable_data = {"|||".join(k): v for k, v in model.items()}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, ensure_ascii=False)
    except Exception:
        pass
        
    return model
