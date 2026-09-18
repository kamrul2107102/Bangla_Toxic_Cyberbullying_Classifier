# External Pretrained Benchmark: BanglaBERT

## 1. Overview & Architecture
**BanglaBERT** (`csebuetnlp/banglabert`) is a state-of-the-art pretrained Bengali language model developed by the **CSE BUET NLP** research group.
- **Model Type**: ELECTRA Discriminator (110M parameters).
- **Pretraining Corpus**: ~27.5 GB of cleaned Bengali text comprising web crawls, Wikipedia, online newspapers, and literature.
- **Pretraining Objective**: Replaced Token Detection (RTD) where a generator network replaces tokens with plausible alternatives and the discriminator predicts whether each token was original or replaced.
- **Hugging Face Checkpoint**: [`csebuetnlp/banglabert`](https://huggingface.co/csebuetnlp/banglabert)

---

## 2. Academic Project Proposal Alignment & Boundary
> [!IMPORTANT]
> **Strict Proposal Separation**:
> The core objective of this academic NLP project is to build and analyze text classification systems from **fundamental algorithms and scratch implementations** (Naive Bayes, TF-IDF + Logistic Regression, Word2Vec Skip-gram, BiLSTM, and Transformer Encoder).
> 
> **BanglaBERT is NOT a submission model in the from-scratch pipeline.**
> It is integrated strictly as an **External Pretrained Reference Benchmark** to answer the research question:
> *"How close can fundamentally engineered, low-resource custom architectures get to a multi-billion-token pretrained foundation model on Bengali cyberbullying detection?"*

---

## 3. Benchmark Pipeline Implementation
The benchmark model is implemented in `src/models/banglabert.py` with the following pipeline:
```text
Raw Comment
  │
  ▼
AutoTokenizer (csebuetnlp/banglabert)
  │
  ▼
BanglaBERT ELECTRA Base (12 Layers, 768 Hidden Dim, 12 Heads)
  │
  ▼
Pooler / [CLS] Sequence Representation
  │
  ▼
Dropout (p=0.2)
  │
  ▼
Linear Classification Head (768 ➔ 5)
  │
  ▼
Sigmoid Multi-Label Output (Hate Speech, Sexist, Threat, Bullying, Toxic)
```

---

## 4. Training Modes

### A. Frozen-Base Mode (Lightweight / CPU / 8 GB RAM)
- All 12 ELECTRA transformer encoder layers are frozen (`param.requires_grad = False`).
- Only the top linear classification head parameters are updated.
- **Memory Footprint**: Extremely low (~1.2 GB VRAM or CPU RAM).
- **Speed**: Fast convergence in 2-3 epochs.

### B. Full Fine-Tuning Mode (Google Colab / GPU)
- The entire ELECTRA backbone and classification head are trained end-to-end.
- Uses **AdamW** optimizer with learning rate `2e-5`, linear warmup, and weight decay `0.01`.
- Provided in `notebooks/train_banglabert_benchmark_colab.ipynb`.

---

## 5. Fair Comparison Safeguards
1. **Identical Partitions**: Uses `artifacts/splits.json` so BanglaBERT evaluates on the exact same 1,991 test samples as the custom models.
2. **Identical Metrics**: Evaluated using the exact same multi-label metrics (Micro/Macro F1, Precision, Recall, Per-label F1).
3. **Threshold Tuning**: Evaluated using validation-set tuned thresholds to prevent test set data leakage.
