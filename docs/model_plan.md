# Model Plan

## 1. Course coverage mapped to the project

The uploaded lab materials cover:
- Lab 1: regex cleaning, tokenization, stop-word removal, stemming, lemmatization, Levenshtein spelling correction.
- Lab 2: Bag-of-Words, TF-IDF, n-gram language modeling and MLE.
- Lab 3: Word2Vec Skip-gram, cosine similarity, Naive Bayes, Logistic Regression, Word2Vec+LR, TF-IDF-weighted embeddings and the word-order limitation.
- Lab 4: PyTorch tensors, pretrained embedding integration, RNN, stacked BiLSTM, sequence labeling, masking, LSTM language modeling, temperature sampling and encoder-decoder Seq2Seq.
- Lab 5: Transformer encoder, positional encoding, self-attention encoder blocks, mean pooling and BCE-with-logits training.

## 2. Recommended experiments

| Model | Representation | Why use it | Status |
|---|---|---|---|
| Multinomial Naive Bayes | word counts | strong, transparent baseline; directly connected to Lab 3 | Core |
| Logistic Regression | manual TF-IDF | strongest classical baseline and directly tied to Lab 2-3 | Core |
| Word2Vec + Logistic Regression | learned dense embeddings | tests Lab 3 semantic representation pipeline | Core |
| BiLSTM | trainable word embeddings | captures sequence/order; directly connected to Lab 4 | Extension |
| Transformer Encoder | learned embeddings + positional encoding | tests modern sequence modeling from Lab 5 | Extension |
| BanglaBERT | pretrained Bengali ELECTRA | useful external benchmark, but violates current proposal constraints | Optional only |

## 3. Why BanglaBERT is not the main model

The project proposal explicitly says the actual learning/classification should not use pretrained/fine-tuned language models and lists NumPy as the numerical foundation. Therefore BanglaBERT belongs in a separate benchmark, not in the primary result table if the proposal constraint is enforced.

## 4. Label design

The canonical outputs are:
- hate_speech
- sexist
- threat
- bullying
- toxic

Neutral is derived when none of these outputs crosses the decision threshold; it is not a sixth independent positive label.

Because source datasets do not annotate the same labels, the merged dataset contains both values and masks. For example, a hate-only dataset supervises `hate_speech` but leaves `sexist`, `threat`, `bullying` and `toxic` unknown. This avoids inventing false negative labels.

## 5. Experimental sequence

1. Prepare and globally deduplicate the four source families.
2. Train TF-IDF + Logistic Regression.
3. Train Multinomial Naive Bayes.
4. Train Word2Vec + Logistic Regression.
5. Compare micro-F1, macro-F1, per-label F1, precision/recall and exact-match on fully annotated samples.
6. Train BiLSTM and Transformer as course-extension models.
7. Evaluate Bangla-ToCo as an external/context-aware test set rather than treating its binary toxic/non-toxic annotation as equivalent to every multi-label category.
8. Optionally run BanglaBERT only as a separately reported benchmark if the instructor allows pretrained models.
