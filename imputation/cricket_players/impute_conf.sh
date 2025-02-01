python impute_w_confidence.py \
    --thresholds 0.7 0.8 0.9 1.0 \
    --model gpt-4o \
    --api_key "sk-pP1nu21e5A9Ucg9bB82c0dC07d334b9aA7A0Bc26F3Eb2f72" \
    --retrieval_results_path "/home/yangchenyu/Data_Imputation/retrieval_results/first_stage/BM25_top100_res_with_score_cricket_players.tsv" \
    --collection_path "/home/yangchenyu/Data_Imputation/data/cricket_players/annotated_data/collection.tsv" \
    --folds_path "/home/yangchenyu/Data_Imputation/data/cricket_players/annotated_data/folds.json" \
    --output_path "/home/yangchenyu/Data_Imputation/imputation/results/cricket_players/gpt-4o_cricket_players_evidence_confidence_BM25_top5.jsonl" \
    --top_k 5 \
    --num_threads 8