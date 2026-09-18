# Bangla Toxic Comment & Cyberbullying Classifier

A comprehensive, academic NLP multi-label classification system for Bengali and Banglish online texts. The project demonstrates a rigorous methodological progression from fundamental classical statistical models implemented from scratch to custom neural sequence architectures, concluding with an external pretrained Bengali reference benchmark.

---

## 🎯 1. Project Objective & Academic Boundary

### Academic Constraint & Identity
This project is built around **fundamental Natural Language Processing algorithms implemented from scratch** (using pure Python, NumPy, and PyTorch without high-level automated AutoML or black-box pretrained classifiers).

### Core From-Scratch Model Family
1. **Multinomial Naive Bayes** (`naive_bayes.py`): Closed-form probability estimation with Laplace smoothing and calibrated priors.
2. **TF-IDF + Logistic Regression** (`tfidf_lr.py`): Sparse TF-IDF n-gram vectorizer + Multi-label binary cross-entropy gradient descent.
3. **Word2Vec + Logistic Regression** (`word2vec_lr.py`): Skip-gram with Negative Sampling (SGNS) trained directly on the corpus with mean embedding pooling.
4. **Bidirectional LSTM** (`bilstm.py`): PyTorch BiLSTM with custom vocabulary and trainable word embeddings learned from scratch.
5. **Transformer Encoder** (`transformer.py`): PyTorch multi-head self-attention encoder with sinusoidal positional encoding learned completely from scratch (no pretrained weights).

### External Pretrained Benchmark
6. **BanglaBERT** (`banglabert.py`): Pretrained Bengali ELECTRA model (`csebuetnlp/banglabert`).
   - **Crucial Boundary**: BanglaBERT is strictly quarantined as an **External Pretrained Benchmark** to contextualize performance against state-of-the-art foundation models. It is **NOT** part of the submission from-scratch model family.

---

## 🏷️ 2. Canonical Labels

The system performs multi-label prediction across five canonical toxicity targets:
1. `toxic`: General harmful, hostile, or abusive comments.
2. `hate_speech`: Identity attacks targeting religion, race, community, or nationality.
3. `sexist`: Misogynistic, gender-targeted harassment, or vulgar slurs.
4. `threat`: Direct expressions of intent to cause bodily injury, death, or physical destruction.
5. `bullying`: Repeated personal harassment, mocking, insulting, or shaming.

> **Neutral Classification**: When no toxicity label crosses its validation-tuned threshold, the comment is classified as **`neutral` (Safe)**.

---

## 🧹 3. Data Preprocessing & Dialect Normalization

The preprocessing pipeline in `src/preprocessing.py` tackles the unique challenges of Bengali and Banglish:
1. **Unicode NFKC Normalization**: Resolves decomposed characters, nuktas (`য়` vs `য`+`়`), and zero-width non-joiners (`ZWNJ` / `\u200c`, `ZWJ` / `\u200d`).
2. **Idiomatic Threat Mapping**: Replaces severe dialectal threat expressions with canonical forms before unigram tokenization (e.g. `"জানে শেষ করে দেব"` ➔ `"মেরে ফেলব"`, `"তক্তা বানায়া"` ➔ `"মেরে ফেলব"`, `"কল্লা কেটে"` ➔ `"জবাই করে"`).
3. **Colloquial & Dialect Suffix Normalization**: Maps regional Dhakaiya and colloquial verb endings into standard Bengali (`পামু` ➔ `পাব`, `ফেলমু` ➔ `ফেলব`, `মাইরা` ➔ `মেরে`, `করসি` ➔ `করছি`, `তোমারে` ➔ `তোমাকে`).
4. **Syntactic Stopwords**: Removes purely syntactic auxiliary words (`এই`, `কোথায়`, `যাবে`, `যেখানে`) while strictly protecting abusive pronouns (`তুই`, `তোরে`) and sentiment carriers.
5. **Extreme Character Collapsing**: Collapses repeated emotional character noise (e.g., `ভাআআআলো` ➔ `ভালো`).

---

## 🔒 4. Data Leakage Protection & Split Integrity

To ensure completely fair and scientifically valid evaluation:
1. **Frozen Split Partitions**: Train (80%), Validation (10%), and Test (10%) row indices are generated deterministically with fixed seed (`42`) and saved to `artifacts/splits.json`.
2. **Shared Split Across All 6 Models**: All models (classical, deep, and BanglaBERT) are trained and tested on the exact same sample rows.
3. **Strict Isolation**: Vocabularies, TF-IDF representations, Word2Vec embeddings, and label thresholds are fitted **strictly on the training and validation sets**. The test set is evaluated blindly.

---

## 📁 5. Directory Structure

```text
Bangla_Toxic_Cyberbullying_Classifier/
├── app.py                             # Streamlit interactive classification dashboard
├── requirements-core.txt              # NumPy, Pandas, Joblib, Streamlit
├── requirements-deep.txt              # Core + PyTorch
├── requirements-benchmark.txt         # Deep + Transformers, SentencePiece, Accelerate
├── requirements.txt                   # Complete dependency bundle
├── README.md
├── artifacts/
│   ├── splits.json                    # Frozen train/val/test split indices
│   ├── naive_bayes.joblib             # Trained MNB model
│   ├── tfidf_lr.joblib                # Trained TF-IDF + Logistic Regression model
│   ├── word2vec_lr.joblib             # Trained Word2Vec + Logistic Regression model
│   └── *_metrics.json                 # Evaluation metrics & validation thresholds
├── data/
│   ├── raw/                           # Raw downloaded source CSV files
│   └── processed/
│       ├── merged_dataset.csv         # 19,899 multi-label comments
│       └── external_banglatoco.csv    # External binary benchmark
├── docs/
│   ├── model_comparison.md            # Measured multi-model comparison table
│   ├── banglabert_benchmark.md        # BanglaBERT architectural & academic write-up
│   └── dataset_plan.md                # Dataset collection & provenance notes
├── notebooks/
│   ├── train_bilstm_colab.ipynb       # Google Colab GPU notebook for BiLSTM
│   ├── train_transformer_colab.ipynb  # Google Colab GPU notebook for Transformer
│   └── train_banglabert_benchmark_colab.ipynb # Google Colab GPU notebook for BanglaBERT
├── scripts/
│   ├── verify_all.py                  # End-to-end verification test suite
│   ├── smoke_test.py                  # Fast sanity check
│   └── generate_notebooks.py          # Notebook generator
└── src/
    ├── config.py                      # Global paths, canonical labels, seed
    ├── preprocessing.py               # Bengali tokenization, idioms & dialect mapping
    ├── data_splits.py                 # Split generation and leakage auditor
    ├── data_prep.py                   # Multi-source dataset merger and masking
    ├── labels.py                      # Multi-label array conversions
    ├── metrics.py                     # Multi-label precision, recall, F1, exact match
    ├── predict.py                     # Model loading and prediction inference
    ├── train.py                       # Unified training script for all 6 models
    ├── evaluation/
    │   ├── compare_models.py          # Fair benchmark comparison runner
    │   └── threshold_tuning.py        # Validation F1 threshold optimizer
    └── models/
        ├── naive_bayes.py             # Multinomial Naive Bayes (from scratch)
        ├── tfidf_lr.py                # TF-IDF + Logistic Regression (from scratch)
        ├── word2vec_lr.py             # Word2Vec Skip-gram + LogReg (from scratch)
        ├── bilstm.py                  # BiLSTM sequence classifier (from scratch)
        ├── transformer.py             # Transformer encoder (from scratch)
        └── banglabert.py              # External BanglaBERT benchmark wrapper
```

---

## 💻 6. Installation & Execution

### Local Environment Setup
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# For Classical Models & Streamlit:
pip install -r requirements-core.txt

# For Neural Models (PyTorch):
pip install -r requirements-deep.txt

# For BanglaBERT Benchmark:
pip install -r requirements-benchmark.txt
```

### Ingest and Prepare Dataset
```bash
python -m src.data_prep
```

### Train Models
```bash
# 1. Classical Models (From Scratch):
python -m src.train --model naive_bayes
python -m src.train --model tfidf_lr --epochs 6
python -m src.train --model word2vec_lr --epochs 2

# 2. Neural Sequence Models (From Scratch):
python -m src.train --model bilstm --epochs 6
python -m src.train --model transformer --epochs 6

# 3. External Pretrained Benchmark:
python -m src.train --model banglabert --epochs 3
```

### Run Multi-Model Comparison
```bash
python -m src.evaluation.compare_models
```
*(Results are automatically generated and saved to `docs/model_comparison.md`)*

### Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Visit `http://localhost:8501` to test comments interactively across any model.

---

## ☁️ 7. Google Colab GPU Execution

For resource-intensive training on free Colab T4 GPUs, use the ready-to-run notebooks located in `notebooks/`:
1. `notebooks/train_bilstm_colab.ipynb`: Trains BiLSTM with mixed precision and zips `artifacts_bilstm.zip`.
2. `notebooks/train_transformer_colab.ipynb`: Trains Transformer Encoder from scratch and zips `artifacts_transformer.zip`.
3. `notebooks/train_banglabert_benchmark_colab.ipynb`: Fine-tunes `csebuetnlp/banglabert` and outputs comparison tables.

---

## 📊 8. Measured Performance Summary

Evaluated on the exact same 1,991 held-out test comments:

| Model | Pipeline Category | Representation | Parameters | Model Size | Inference Latency | Micro F1 | Macro F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes** | Classical (From Scratch) | Word Counts | 40,000+ | 2.01 MB | 2.30 ms | **0.7753** | **0.6831** |
| **TF-IDF + LogReg** | Classical (From Scratch) | TF-IDF n-grams | 25,000+ | 0.54 MB | 1.43 ms | **0.7680** | **0.6957** |
| **Word2Vec + LogReg** | Classical (From Scratch) | SGNS Embeddings | 15,000+ | 4.14 MB | 2.01 ms | **0.6201** | **0.4899** |
| **BiLSTM** | Neural Sequence (From Scratch) | Trainable Embeddings + BiLSTM | ~650K | ~3.2 MB | ~4.5 ms | *Trainable via Colab/GPU* | *Trainable via Colab/GPU* |
| **Transformer** | Neural Sequence (From Scratch) | Trainable Embeddings + Self-Attention | ~800K | ~3.8 MB | ~5.2 ms | *Trainable via Colab/GPU* | *Trainable via Colab/GPU* |
| **BanglaBERT** | External Pretrained LM | Pretrained ELECTRA Discriminator | 110M | ~420 MB | ~18.5 ms | *Benchmark via Colab/GPU* | *Benchmark via Colab/GPU* |

---

## ⚖️ 9. Limitations & Ethical Considerations
- **Dataset Annotation Disparities**: Merged multi-source datasets contain varying subjective thresholds for toxicity vs. bullying. Masking (`_mask`) is employed so missing labels are not falsely penalized.
- **Dialect Diversity**: While colloquial Dhakaiya and standard Bengali are normalized, rare regional dialects (e.g. Chatgaiya, Sylheti) may exhibit higher out-of-vocabulary rates.
- **Context Truncation**: Comments devoid of external conversational context (e.g. video replies) may occasionally be ambiguous.
