#!/bin/bash
MODEL_PATH=/home/yangchenyu/Data_Imputation/Reranker/output/bert/generated_show_movie_pretrained_epoch20_lr_5e-5  # 你训练好的模型路径
PRED_PATH="/home/yangchenyu/Data_Imputation/Reranker/data/show_movie/show_movie.test.jsonl"      # 预测数据的路径
PRED_ID_FILE="/home/yangchenyu/Data_Imputation/Reranker/data/show_movie/pred_ids.txt"          # 包含query_id和passage_id的文件
OUTPUT_DIR="./results"                    # 输出目录
SCORE_PATH="$OUTPUT_DIR/generated_show_movie_pretrained_epoch20_lr_5e-5.test.scores.txt"      # 评分输出文件

CUDA_VISIBLE_DEVICES=2 python train.py \
  --model_name_or_path $MODEL_PATH \
  --do_predict \
  --pred_path $PRED_PATH \
  --pred_id_file $PRED_ID_FILE \
  --rank_score_path $SCORE_PATH \
  --output_dir $OUTPUT_DIR \
  --max_len 512 \
  --per_device_eval_batch_size 256 \
  --dataloader_num_workers 4