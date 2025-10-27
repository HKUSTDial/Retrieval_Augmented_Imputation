#!/bin/bash
# Script for making predictions with a trained reranker model
# This script loads a trained model and generates ranking scores for query-passage pairs

# Path to the trained model directory
MODEL_PATH=YOUR_TRAINED_MODEL_PATH

# Path to the prediction data file (JSONL format)
PRED_PATH="YOUR_PREDICTION_DATA_PATH"

# Path to the file containing query_id and passage_id pairs (one per line, space-separated)
PRED_ID_FILE="YOUR_PRED_ID_FILE_PATH"

# Output directory for results
OUTPUT_DIR="./results"

# Path to the output score file
SCORE_PATH="$OUTPUT_DIR/your_model_name.test.scores.txt"

# Path to the qrel file
QREL_PATH="YOUR_QREL_FILE_PATH"

# Run prediction
CUDA_VISIBLE_DEVICES=2 python train.py \
  --model_name_or_path $MODEL_PATH \
  --do_predict \
  --pred_path $PRED_PATH \
  --pred_id_file $PRED_ID_FILE \
  --rank_score_path $SCORE_PATH \
  --output_dir $OUTPUT_DIR \
  --qrel_path $QREL_PATH \
  --max_len 512 \
  --per_device_eval_batch_size 256 \
  --dataloader_num_workers 4