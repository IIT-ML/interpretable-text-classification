#!/usr/bin/env bash

# python3 train_baseline.py \
# --word_len '300' \
# --dataset "imdb" & 
python3 train_baseline.py \
--word_len '300' \
--dataset "arxiv" & 
# python3 train_baseline.py \
# --word_len '300' \
# --dataset "agnews" & 
