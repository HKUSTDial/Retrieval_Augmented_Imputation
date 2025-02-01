#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=3 python train.py \
    --output_dir ./output/bert/cricket_players_pretrained_epoch30_lr_5e-5 \
    --model_name_or_path /home/yangchenyu/pre-trained-models/Luyu/bert-base-mdoc-bm25 \
    --train_path data/cricket_players/generated_cricket_players.train.jsonl \
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
