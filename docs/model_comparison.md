# Fair Multi-Model Comparison: Bangla Toxic & Cyberbullying Classification

**Test Set Samples**: 1991 (held-out stratified partition strictly isolated from training)

> **Academic Framework Note**:
> All core models (Naive Bayes, TF-IDF + LR, Word2Vec + LR, BiLSTM, Transformer) are implemented and trained **from scratch**.
> **BanglaBERT** is included exclusively as an **External Pretrained Benchmark** to contextualize how fundamental algorithmic implementations compare with large pretrained language models.

## 1. Overall Performance & Efficiency Comparison

| Model | Pipeline Category | Representation | Parameters | Model Size | Inference / Sample | Micro F1 | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naive Bayes | Classical (From Scratch) | Bag of Words (Count) | N/A | 2.01 MB | 1.00 ms | **0.7753** | **0.6831** |
| TF-IDF + Logistic Regression | Classical (From Scratch) | Sparse TF-IDF n-grams | N/A | 0.54 MB | 0.69 ms | **0.7680** | **0.6957** |
| Word2Vec + Logistic Regression | Classical (From Scratch) | SGNS Learned Embeddings (Mean Pooled) | N/A | 4.14 MB | 0.86 ms | **0.6847** | **0.5899** |

## 2. Per-Label F1 Scores

| Model | Hate_speech | Sexist | Threat | Bullying | Toxic |
| --- | --- | --- | --- | --- | --- |
| Naive Bayes | 0.7082 | 0.5687 | 0.5868 | 0.6520 | 0.9000 |
| TF-IDF + Logistic Regression | 0.7061 | 0.6091 | 0.6224 | 0.6532 | 0.8877 |
| Word2Vec + Logistic Regression | 0.5738 | 0.4691 | 0.5000 | 0.5553 | 0.8512 |

## 3. Key Observations

1. **Zero Data Leakage**: All models use frozen train/validation/test index splits (`artifacts/splits.json`). Preprocessing and vocabularies are learned solely on the training partition.
2. **Threshold Optimization**: Detection thresholds for all models are determined exclusively on validation data using F1-maximization, avoiding test-set overfitting.
3. **Inference Latency vs. Capacity Trade-Off**: Classical linear models offer near-instantaneous CPU inference (<1 ms/sample) while sequence models capture sequential context and word order.
