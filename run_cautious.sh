#!/usr/bin/env bash


python3 train.py \
--word_len '200' \
--dataset "arxiv" \
--train_mode 1 \
--epochs 1 \
--batch_size 1 
# python3 train.py \
# --word_len '200' \
# --dataset "agnews" \
# --train_mode 2 \
# --epochs 1 \
# --batch_size 1 &
# python3 train.py \
# --word_len '200' \
# --dataset "agnews" \
# --train_mode 3 \
# --epochs 1 \
# --batch_size 1 &
# python3 train.py \
# --gpu \
# --dataset "imdb" \
# --train_mode 1 \
# --epochs 1 \
# --batch_size 1 && \
# python3 train.py \
# --gpu \
# --dataset "imdb" \
# --train_mode 1 \
# --epochs 1 \
# --batch_size 1 && \
# python3 train.py \
# --gpu \
# --dataset "imdb" \
# --train_mode 1 \
# --epochs 1 \
# --batch_size 1 && \