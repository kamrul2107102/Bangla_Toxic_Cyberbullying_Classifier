from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

LABELS = ["hate_speech", "sexist", "threat", "bullying", "toxic"]
SEED = 42
MAX_VOCAB = 12000
MAX_W2V_VOCAB = 8000
EMBED_DIM = 64
MAX_LEN = 64
