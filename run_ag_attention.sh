#!/usr/bin/env bash

python3 train_hierarchical.py \
--dataset "imdb" \
--model 'HN' & \
python3 train_hierarchical.py \
--dataset "imdb" \
--model 'HAN' & \