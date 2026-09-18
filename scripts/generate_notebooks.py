import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def create_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "language_info": {"name": "python"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


# ==========================================
# 1. BiLSTM Colab Notebook
# ==========================================
bilstm_cells = [
    md_cell("""# 🚀 Train BiLSTM Multi-Label Classifier from Scratch (Google Colab GPU)
This notebook trains the **BiLSTM Multi-Label Bengali Toxic Comment Classifier** using PyTorch on Google Colab (Free T4 GPU).

### Key Highlights:
- **No pretrained word embeddings**: Word embeddings are learned strictly from scratch on the Bengali training dataset.
- **Zero Data Leakage**: Loads and respects the exact `splits.json` indices used by the classical baseline models.
- **Hardware Acceleration**: Automatic CUDA / Mixed Precision support.
- **Artifact Export**: Saves weights, vocabulary, config, and thresholds, then zips them for local deployment in the Streamlit app.
"""),
    code_cell("""# Step 1: Check GPU & Environment
!pip install -q torch numpy pandas joblib
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
"""),
    code_cell("""# Step 2: Upload colab_package.zip (Located in your project root)
import os
from google.colab import files

if not os.path.exists("src"):
    print("📁 Please upload 'colab_package.zip' (located in your local project root):")
    uploaded = files.upload()
    for fname in uploaded.keys():
        if fname.endswith(".zip"):
            !unzip -q {fname}
            print(f"✅ Extracted {fname} successfully!")
            break
else:
    print("✅ Project source files already present!")
"""),
    code_cell("""# Step 3: Run BiLSTM GPU Training
# Train for 8 epochs on GPU with batch size 64:
!python -m src.train --model bilstm --epochs 8
"""),
    code_cell("""# Step 4: Zip & Automatically Download Artifacts
import shutil
from google.colab import files

shutil.make_archive("artifacts_bilstm", "zip", "artifacts")
print("✅ Created 'artifacts_bilstm.zip'!")
print("⬇️ Initiating automatic download to your computer...")
files.download("artifacts_bilstm.zip")
""")
]

# ==========================================
# 2. Transformer from Scratch Colab Notebook
# ==========================================
transformer_cells = [
    md_cell("""# 🚀 Train Learned-from-Scratch Transformer Encoder (Google Colab GPU)
This notebook trains a **Lightweight Transformer Encoder Multi-Label Classifier** learned strictly from scratch on Bengali text.

### Architectural Blueprint:
- Token IDs ➔ Trainable Token Embedding (128-dim) ➔ Sinusoidal Positional Encoding
- PyTorch `TransformerEncoder` (2 Layers, 4 Attention Heads, 256 Feedforward Dim)
- Masked Sequence Average Pooling ➔ Fully Connected Classification Head
- Binary Cross Entropy with Logits Loss (`BCEWithLogitsLoss`)
- Zero pretrained Hugging Face transformer weights (100% academic from-scratch integrity).
"""),
    code_cell("""# Step 1: Check GPU & Environment
!pip install -q torch numpy pandas joblib
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
"""),
    code_cell("""# Step 2: Upload colab_package.zip (Located in your project root)
import os
from google.colab import files

if not os.path.exists("src"):
    print("📁 Please upload 'colab_package.zip' (located in your local project root):")
    uploaded = files.upload()
    for fname in uploaded.keys():
        if fname.endswith(".zip"):
            !unzip -q {fname}
            print(f"✅ Extracted {fname} successfully!")
            break
else:
    print("✅ Project source files already present!")
"""),
    code_cell("""# Step 3: Train Transformer from Scratch on GPU
# Runs with gradient clipping and validation-based threshold optimization
!python -m src.train --model transformer --epochs 8
"""),
    code_cell("""# Step 4: Zip & Automatically Download Artifacts
import shutil
from google.colab import files

shutil.make_archive("artifacts_transformer", "zip", "artifacts")
print("✅ Created 'artifacts_transformer.zip'!")
print("⬇️ Initiating automatic download to your computer...")
files.download("artifacts_transformer.zip")
""")
]

# ==========================================
# 3. BanglaBERT Benchmark Colab Notebook
# ==========================================
banglabert_cells = [
    md_cell("""# 🧪 BanglaBERT External Pretrained Benchmark (Google Colab GPU)
This notebook executes the **External Pretrained Benchmark** using **BanglaBERT** (`csebuetnlp/banglabert`).

### Academic Boundary Notice:
> **IMPORTANT**: BanglaBERT is a pretrained Bengali ELECTRA model (110M parameters).
> It is **NOT** part of the proposal-compliant from-scratch model family.
> It is included strictly as an external reference benchmark to contextualize how custom architectures compare against a state-of-the-art pretrained Bengali language model.
"""),
    code_cell("""# Step 1: Install Hugging Face Transformers & Benchmark Dependencies
!pip install -q transformers sentencepiece accelerate torch pandas numpy joblib
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
"""),
    code_cell("""# Step 2: Upload colab_package.zip (Located in your project root)
import os
from google.colab import files

if not os.path.exists("src"):
    print("📁 Please upload 'colab_package.zip' (located in your local project root):")
    uploaded = files.upload()
    for fname in uploaded.keys():
        if fname.endswith(".zip"):
            !unzip -q {fname}
            print(f"✅ Extracted {fname} successfully!")
            break
else:
    print("✅ Project source files already present!")
"""),
    code_cell("""# Step 3: Download BanglaBERT Checkpoint & Fine-Tune
# Fine-tune on the EXACT same train split and evaluate on the exact same test split:
!python -m src.train --model banglabert --epochs 3
"""),
    code_cell("""# Step 4: Generate Unified Multi-Model Comparison Table
!python -m src.evaluation.compare_models
"""),
    code_cell("""# Step 5: Zip & Automatically Download Benchmark Artifacts
import shutil
from google.colab import files

shutil.make_archive("artifacts_banglabert", "zip", "artifacts")
print("✅ Saved 'artifacts_banglabert.zip'!")
print("⬇️ Initiating automatic download to your computer...")
files.download("artifacts_banglabert.zip")
""")
]

(NOTEBOOKS_DIR / "train_bilstm_colab.ipynb").write_text(json.dumps(create_notebook(bilstm_cells), indent=2), encoding="utf-8")
(NOTEBOOKS_DIR / "train_transformer_colab.ipynb").write_text(json.dumps(create_notebook(transformer_cells), indent=2), encoding="utf-8")
(NOTEBOOKS_DIR / "train_banglabert_benchmark_colab.ipynb").write_text(json.dumps(create_notebook(banglabert_cells), indent=2), encoding="utf-8")

print("Regenerated all 3 Colab notebooks in notebooks/")
