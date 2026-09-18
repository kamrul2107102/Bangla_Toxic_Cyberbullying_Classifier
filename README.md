# Bangla Toxic Comment & Cyberbullying Classifier

A course-aligned multi-label Bengali/Banglish toxicity classifier built around the concepts taught in the uploaded NLP lab materials.

## What is in the project?

### Proposal-compliant core
- Bengali/Banglish preprocessing inspired by Lab 1.
- Manual TF-IDF + Logistic Regression from scratch.
- Manual Multinomial Naive Bayes from scratch.
- Word2Vec Skip-gram with negative sampling + Logistic Regression, implemented with NumPy.
- Dataset masks so unknown labels are not treated as false negatives.

### Course-extension models
- PyTorch stacked BiLSTM.
- PyTorch Transformer encoder with positional encoding.

### Optional benchmark
- BanglaBERT (`csebuetnlp/banglabert`) is provided only as an optional benchmark because your project proposal currently prohibits pretrained/fine-tuned language models for the actual classifier.

## Directory

```text
Bangla_Toxic_Cyberbullying_Classifier/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                # put downloaded CSV files here
│   └── processed/
├── artifacts/              # trained models + metrics
├── docs/
│   ├── model_plan.md
│   ├── dataset_plan.md
│   └── banglabert_optional.py
├── scripts/
│   ├── smoke_test.py
│   └── train_all.sh
└── src/
    ├── config.py
    ├── data_prep.py
    ├── labels.py
    ├── metrics.py
    ├── preprocessing.py
    ├── predict.py
    ├── train.py
    └── models/
        ├── tfidf_lr.py
        ├── naive_bayes.py
        ├── word2vec_lr.py
        └── deep_models.py
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements-core.txt
# For BiLSTM/Transformer/BanglaBERT extensions:
pip install -r requirements-deep.txt
```

## 1. Add datasets

Download the datasets listed in `docs/dataset_plan.md` and place their CSV files into `data/raw/`.

Because Kaggle/Mendeley files can have slightly different filenames, the ingestion script detects datasets from filename keywords and common column names. If a downloaded file uses a different schema, add a small adapter in `src/data_prep.py`.

## 2. Prepare the unified dataset

```bash
python -m src.data_prep
```

This creates:

```text
data/processed/merged_dataset.csv
data/processed/all_sources_dataset.csv
data/processed/external_banglatoco.csv   # when Bangla-ToCo is present
```

The prepared table contains:

```text
text, source, context_text,
hate_speech, sexist, threat, bullying, toxic,
hate_speech_mask, sexist_mask, threat_mask, bullying_mask, toxic_mask
```

A mask value of `0` means the source did not annotate that label; that output is ignored during model fitting and evaluation.

## 3. Run the small smoke test

```bash
python scripts/smoke_test.py
```

## 4. Train the classical models

```bash
python -m src.train --model tfidf_lr --epochs 8
python -m src.train --model naive_bayes
python -m src.train --model word2vec_lr --epochs 2
```

Deep models:

```bash
python -m src.train --model bilstm --epochs 5
python -m src.train --model transformer --epochs 5
```

For GPU training, install a matching PyTorch build for your CUDA version.

## 5. External Bangla-ToCo evaluation

```bash
python -m src.evaluate_external --model tfidf_lr --data data/processed/external_banglatoco.csv
```

This evaluates only the `toxic` label because Bangla-ToCo is annotated as Toxic/Non-Toxic rather than the full five-label schema.

## 6. Launch the interface

```bash
streamlit run app.py
```

The UI lets you choose a trained model, enter Bengali/Banglish text, see each label probability and inspect the preprocessed token sequence.

## Evaluation

The project reports:
- micro precision/recall/F1
- macro F1
- per-label precision/recall/F1
- exact-match accuracy on rows where all five canonical labels are genuinely annotated

Accuracy should not be the only metric because the dataset is multi-label and heterogeneous.

## Model comparison table for the report

| Experiment | Representation | Sequence aware? | Pretrained? |
|---|---|---:|---:|
| MNB | word counts | No | No |
| TF-IDF + LR | TF-IDF | No | No |
| Word2Vec + LR | dense word vectors | No (mean pooling) | No |
| BiLSTM | trainable embeddings | Yes | No |
| Transformer | trainable embeddings + position | Yes | No |
| BanglaBERT | pretrained Bengali ELECTRA | Yes | **Yes** (optional benchmark) |

## Important methodological note

The canonical labels are `hate_speech`, `sexist`, `threat`, `bullying`, and `toxic`. `neutral` is derived when no positive label is predicted.

The heterogeneous-source merge is intentionally conservative: labels absent from a source are marked unknown rather than automatically set to zero. Bangla-ToCo is kept separately as an external context-aware test source because it has a binary Toxic/Non-Toxic annotation rather than the full five-label schema.

## Viva points

- Lab 1 -> preprocessing and Levenshtein.
- Lab 2 -> BoW/TF-IDF and why common words get lower weight.
- Lab 3 -> Word2Vec Skip-gram, cosine similarity, Naive Bayes, Logistic Regression, TF-IDF weighted embeddings and the word-order limitation.
- Lab 4 -> RNN/BiLSTM, padding, masking, BCEWithLogitsLoss and sequence modeling.
- Lab 5 -> self-attention/Transformer encoder, positional encoding, mean pooling.
