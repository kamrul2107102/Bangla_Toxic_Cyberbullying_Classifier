#!/usr/bin/env bash
set -e
python -m src.data_prep
python -m src.train --model tfidf_lr --epochs 8
python -m src.train --model naive_bayes
python -m src.train --model word2vec_lr --epochs 2
# Deep models are optional; run on a GPU machine for faster experimentation.
# python -m src.train --model bilstm --epochs 5
# python -m src.train --model transformer --epochs 5
