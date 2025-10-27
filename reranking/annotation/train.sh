#!/bin/bash

python construct_training_groups.py \
    --collection /home/yangchenyu/Data_Imputation/data/show_movie/annotated_data/collection.tsv \
    --qrels /home/yangchenyu/Data_Imputation/data/show_movie/annotated_data/qrels.tsv \
    --folds /home/yangchenyu/Data_Imputation/data/show_movie/annotated_data/folds.json \
    --queries /home/yangchenyu/Data_Imputation/data/show_movie/annotated_data/queries.tsv \
    --retrieved_results /home/yangchenyu/Retrieval_Augmented_Imputation/retriever/retrieval_results/Siamese_show_movie_top100_res_with_score.tsv \
    --api_key "sk-jAAFoKRmOVa8oxQj3aF19e9dC5F4482683D7751c2bCb0595" \
    --api_base "https://vip.yi-zhan.top/v1" \
    --model "gpt-4o" \
    --max_annotations_per_query 30 \
    --min_negative_samples 15 \
    --min_positive_samples 1 \
    --output_path "./results/dimension_evaluation_results.jsonl" \
    --generate_training \
    --training_output_path "../data/show_movie/training_data.jsonl" \
    --negative_samples_per_group 15
