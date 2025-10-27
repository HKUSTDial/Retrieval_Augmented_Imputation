#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Run evaluation
python evaluate.py \
  --model_name='siamese' \
  --dataset_name='show_movie' \
  --num_retrieved=100 \
  --save_model_dir='./model_checkpoints/siamese_model' \
  --default_path='./pretrained_models/bert-base-uncased' \
  --temp_index_path='./index' \
  --data_path='../data/show_movie' 