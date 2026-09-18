# Analysis of the uploaded NLP Lab materials

## Lab 1 - Text preprocessing and spelling correction
Taught concepts:
- Regular expressions for HTML/special-character cleanup and whitespace normalization.
- Word tokenization.
- Stop-word removal.
- Porter stemming.
- Lemmatization with POS awareness.
- Levenshtein edit distance with dynamic programming.
- Simple spelling correction by nearest dictionary word.

Project use:
- Keep the regex/tokenization idea, but make the character rules Bengali-aware.
- Do not blindly remove English stopwords from Bangla/Banglish data.
- Keep a Levenshtein module as an optional spelling-normalization experiment.

## Lab 2 - Text representation and n-gram language modeling
Taught concepts:
- Bag of Words.
- TF-IDF and the TF/IDF equations.
- N-gram language models and the Markov assumption.
- Maximum Likelihood Estimation for bigrams/trigrams.
- Shannon next-word prediction and text generation.

Project use:
- TF-IDF is a primary classical representation.
- N-gram language modeling is not the core classifier because the project target is label prediction, not next-word generation.

## Lab 3 - Word embeddings and foundational classifiers
Taught concepts:
- Word2Vec Skip-gram from scratch using a center word and context words.
- Softmax, cross-entropy, gradient descent.
- Cosine similarity and vector arithmetic.
- Naive Bayes with Laplace smoothing and log probabilities.
- Logistic Regression with sigmoid and gradient descent.
- Word2Vec + Logistic Regression using mean document vectors.
- TF-IDF weighted word embeddings.
- Demonstration that mean/TF-IDF aggregation loses word order.

Project use:
- Multinomial Naive Bayes baseline.
- TF-IDF + Logistic Regression baseline.
- Word2Vec + Logistic Regression.
- The word-order limitation motivates the sequence-model extensions.

## Lab 4 - PyTorch sequence models
Taught concepts:
- Tensors and CPU/GPU device management.
- `nn.Module`, embeddings and training loops.
- `nn.RNN`, `nn.LSTM`, stacked BiLSTM.
- Padding/truncation and `ignore_index` loss masking.
- Sequence classification vs. token-level sequence labeling.
- Batch matrix multiplication for attention mechanics.
- Autoregressive LSTM language modeling and temperature sampling.
- Encoder-decoder sequence-to-sequence translation with teacher forcing.

Project use:
- Stacked BiLSTM classifier as a course-aligned extension model.
- Mask-aware BCE loss handles the heterogeneous multi-label dataset.

## Lab 5 - Transformer
Taught concepts:
- Token embeddings.
- Sinusoidal positional encoding.
- Transformer encoder layers with self-attention.
- Padding masks.
- Mean pooling into one sequence representation.
- Linear classification head.
- BCEWithLogitsLoss.
- Tensor shape flow through the encoder.

Project use:
- Small Transformer encoder classifier as a course-aligned extension model.

## Final model ladder

1. Naive Bayes -> transparent probabilistic baseline.
2. TF-IDF + Logistic Regression -> strong classical baseline.
3. Word2Vec + Logistic Regression -> dense semantic representation.
4. BiLSTM -> sequence-aware model.
5. Transformer -> attention-based sequence-aware model.
6. BanglaBERT -> optional external benchmark only, because the project proposal explicitly prohibits pretrained/fine-tuned language models in the actual classifier.
