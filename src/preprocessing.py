import re
import unicodedata
from typing import List

# Bengali Unicode block + Latin/ASCII + digits.
TOKEN_RE = re.compile(r"[\u0980-\u09FF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
REPEAT_RE = re.compile(r"(.)\1{2,}")

# Neutral Bengali grammatical stop words (purely syntactic, relative, auxiliary)
BANGLA_STOPWORDS = {
    "এই", "সেই", "একটি", "একটা", "কি", "কী", "কেন", "কোথায়", "কোথায়", "কিভাবে",
    "কখন", "কোন", "কোনো", "যে", "সে", "তিনি", "তারা", "তাদের", "তার", "তাকে",
    "আমরা", "আমাদের", "তোমরা", "তোমাদের", "আপনি", "আপনার", "আপনারা",
    "হবে", "হলে", "হলো", "হচ্ছে", "হয়েছে", "হয়ে", "হয়ে", "আছে", "ছিল", "থাকবে",
    "যাবে", "যেতে", "থেকে", "দিয়ে", "করে", "করতে", "করা", "এবং", "বা",
    "কিন্তু", "যদি", "তবে", "তাহলে", "আর", "তো", "ও", "না", "নি", "নাই", "নেই",
    "যেখানে", "সেখানে", "এখানে", "কোথাও", "কখনো", "কাউকে", "কারো", "যাকে", "যাদের",
    "এসব", "ওসব", "নাকি", "যেন"
}

# Multi-word phrase normalizations (mapping colloquial threat/abuse idioms to canonical forms)
PHRASE_MAPPINGS = [
    # Death threats & violent threats idioms
    (re.compile(r"জানে\s+শেষ(?:\s+করে\s+দেব|\s+করে\s+দেবো|\s+করে\s+দেমু)?", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"জান(?:\s+থেকে)?\s+মেরে", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"তক্তা\s+(?:বানায়া|বানায়া|বানিয়ে|বানিয়ে|বানামু|বানাব)", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"মাটিতে\s+পুতে(?:\s+ফেলব|\s+ফেলমু|\s+দেব|\s+দেমু)?", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"পুতে\s+(?:ফেলব|ফেলমু|দেব|দেমু)", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"উড়(?:াই|ি)য়া\s+(?:দেব|দেমু|ফেলব|ফেলমু)", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"উড়িয়ে\s+(?:দেব|দেবো|ফেলব)", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"(?:কল্লা|গলা)\s+(?:কেটে|কাটা)", re.IGNORECASE), "জবাই করে"),
    (re.compile(r"(?:টুকরো|টুকরা)\s+টুকরো", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"কচুকাটা\s+কর", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"হাত\s*(?:পা|ঠ্যাং)\s*(?:ভেঙ্গে|ভেঙে)", re.IGNORECASE), "মারব মেরে"),
    (re.compile(r"পিঠের\s+চামড়া", re.IGNORECASE), "মারব মেরে"),
    (re.compile(r"মাইরা\s+ফেল(?:ব|বো|মু)", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"খুন\s+করে\s+ফেল(?:ব|বো|মু)", re.IGNORECASE), "মেরে ফেলব"),
    (re.compile(r"জবাই\s+করে\s+ফেল(?:ব|বো|মু)", re.IGNORECASE), "জবাই করে ফেলব"),
]

# Exact token substitutions for colloquial / dialectal terms
TOKEN_MAPPINGS = {
    # Participles
    "মাইরা": "মেরে",
    "ধইরা": "ধরে",
    "খাইয়া": "খেয়ে",
    "খাইয়া": "খেয়ে",
    "বানায়া": "বানিয়ে",
    "বানায়া": "বানিয়ে",
    "ফালাইয়া": "ফেলে",
    "ফালাইয়া": "ফেলে",
    "ফালায়": "ফেলে",
    "উড়াইয়া": "উড়িয়ে",
    "উড়াইয়া": "উড়িয়ে",
    "কোপায়া": "কোপিয়ে",
    "কোপায়া": "কোপিয়ে",
    "বসাইয়া": "বসিয়ে",
    "বসায়া": "বসিয়ে",
    # Dialect pronouns
    "আমারে": "আমাকে",
    "তোমারে": "তোমাকে",
    "তোরারে": "তোমাদের",
    "তোগো": "তোমাদের",
    "আমগো": "আমাদের",
    # Dialect verbs
    "পামু": "পাব",
    "ফেলমু": "ফেলব",
    "মারমু": "মারব",
    "করমু": "করব",
    "যামু": "যাব",
    "দিমু": "দেব",
    "দেমু": "দেব",
    "খামু": "খাব",
    "লমু": "নেব",
    "কমু": "বলব",
    "বানামু": "বানাব",
    "পিটামু": "মারব",
    "পিটাব": "মারব",
    "পিটাবো": "মারব",
    "পিটাইয়া": "মেরে",
    "পিটিয়ে": "মেরে",
    "করসি": "করছি",
    "গেসি": "গেছি",
    "খাইসি": "খেয়েছি",
    "মারসি": "মেরেছি",
    "করসে": "করেছে",
    "গিসে": "গিয়েছে",
    "মারসে": "মেরেছে",
    # Threat / violent slang words
    "তক্তা": "মেরে",
}

EXCLUDE_MU = {"আম্মু", "হুকুম", "মরহুম", "মৌসুম", "মাসুম", "মুহুর্ত"}


def normalize_text(text: str) -> str:
    """Clean and normalize Bengali/Banglish text, resolving dialectal idioms and noise."""
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKC", text)
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("\u200c", "").replace("\u200d", "")
    # Collapse extreme character repetition
    text = REPEAT_RE.sub(r"\1\1", text)
    # Apply multi-word phrase mappings (threat/abuse idioms)
    for pattern, repl in PHRASE_MAPPINGS:
        text = pattern.sub(repl, text)
    # Keep Bengali/Latin/digits/whitespace; remove punctuation noise
    text = re.sub(r"[^\u0980-\u09FFA-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    norm = normalize_text(text)
    raw_toks = TOKEN_RE.findall(norm)
    final_toks = []
    for t in raw_toks:
        # Token-level colloquial mappings
        if t in TOKEN_MAPPINGS:
            t = TOKEN_MAPPINGS[t]
        elif t.endswith("মু") and len(t) >= 4 and t not in EXCLUDE_MU:
            t = t[:-2] + "ব"
        if remove_stopwords and t in BANGLA_STOPWORDS:
            continue
        final_toks.append(t)
    return final_toks


def preprocess(text: str) -> List[str]:
    return tokenize(text, remove_stopwords=True)


def join_tokens(text: str) -> str:
    return " ".join(preprocess(text))
