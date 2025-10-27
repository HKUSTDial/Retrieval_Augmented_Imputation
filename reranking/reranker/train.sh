#!/bin/bash
# Script for training a reranker model
# This script configures hyperparameters and paths for training

# Model: Use the pretrained BERT reranker from Luyu/bert-base-mdoc-bm25
# Download from: https://huggingface.co/Luyu/bert-base-mdoc-bm25
# You can download and use the model by setting the path below to your local model directory
# Or use the model name directly: 'Luyu/bert-base-mdoc-bm25' (requires internet connection)

export CUDA_LAUNCH_BLOCKING=1

# GPU device ID to use (change based on available GPUs)
CUDA_VISIBLE_DEVICES=3 python train.py \
    --output_dir ./output/bert/your_model_name \
    --model_name_or_path Luyu/bert-base-mdoc-bm25 \
    --train_path YOUR_TRAINING_DATA_PATH \
    --max_len 512 \
    --per_device_train_batch_size 1 \
    --train_group_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --num_train_epochs 30 \
    --save_steps 2000 \
    --seed 42 \
    --do_train \
    --logging_steps 20 \
    --overwrite_output_dir
