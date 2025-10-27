#!/bin/bash

python construct_training_groups.py \
    --collection /path/to/collection.tsv \
    --qrels /path/to/qrels.tsv \
    --folds /path/to/folds.json \
    --queries /path/to/queries.tsv \
    --retrieved_results /path/to/retrieved_results.tsv \
    --api_key "your-openai-api-key" \
    --api_base "https://api.openai.com/v1" \
    --model "gpt-4o" \
    --max_annotations_per_query 30 \
    --min_negative_samples 15 \
    --min_positive_samples 1 \
    --output_path "./results/dimension_evaluation_results.jsonl" \