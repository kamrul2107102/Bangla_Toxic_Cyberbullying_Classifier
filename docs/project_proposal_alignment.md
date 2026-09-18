# Proposal Alignment

The current proposal describes a from-scratch Bengali multi-label toxic/cyberbullying system with custom tokenizer, vocabulary, feature representation, neural network, manually implemented loss/backpropagation and multi-label thresholds, while explicitly avoiding pretrained/fine-tuned language models and high-level NLP/ML learning APIs for the core. See the attached proposal for the exact project framing.

This repository therefore separates the project into:
1. a proposal-compliant NumPy core (manual preprocessing, TF-IDF, NB, LR, Word2Vec + LR), and
2. a PyTorch course-extension track (BiLSTM/Transformer), because the lab materials taught these architectures but the proposal currently lists NumPy-only core learning.

If the instructor requires strict proposal compliance, report the NumPy models as the main system and present BiLSTM/Transformer/BanglaBERT only as comparative extensions or future work.
