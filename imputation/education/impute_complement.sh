python impute_complement.py \
    --thresholds 0.7 0.8 0.9 1.0 \
    --model gpt-4o \
    --retriever reranker \
    --data_path "/home/yangchenyu/Data_Imputation/data/education/" \
    --api_key "sk-pP1nu21e5A9Ucg9bB82c0dC07d334b9aA7A0Bc26F3Eb2f72" \
    --retrieval_results_path "/home/yangchenyu/Data_Imputation/Reranker/results/final/generated_education_pretrained_epoch20_lr_5e-5.test.scores.txt" \
    --collection_path "/home/yangchenyu/Data_Imputation/data/education/annotated_data/collection.tsv" \
    --folds_path "/home/yangchenyu/Data_Imputation/data/education/annotated_data/folds.json" \
    --output_path "/home/yangchenyu/Data_Imputation/imputation/results/education/gpt-4o_education_evidence_confidence_reranker_top5.jsonl" \
    --top_k 5 \
    --num_threads 8
    