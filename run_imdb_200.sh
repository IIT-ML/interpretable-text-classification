#!/usr/bin/env bash

python3 train_imdb.py \
--word_len '200' \
--dataset "imdb" \
--train_mode 1 \
--epochs 1000 \
--batch_size 8 && \
python3 train_imdb.py \
--word_len '200' \
--dataset "imdb" \
--train_mode 2 \
--epochs 1000 \
--batch_size 8 && \
python3 train_imdb.py \
--word_len '200' \
--dataset "imdb" \
--train_mode 3 \
--epochs 1000 \
--batch_size 8