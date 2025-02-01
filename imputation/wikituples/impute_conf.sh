python impute_w_confidence.py \
    --api_key "sk-pP1nu21e5A9Ucg9bB82c0dC07d334b9aA7A0Bc26F3Eb2f72" \
    --num_threads 32 \
    --model gpt-4o-mini \
    --missing_tables_path /home/yangchenyu/Data_Imputation/data/wikituples/missing_tables.jsonl \
    --retrieval_results_path /home/yangchenyu/Data_Imputation/Reranker/results/final/generated_pretrained_bert_wikituples_epoch30.test.scores.txt \
    --collection_path ../data/wikituples/final_data/collection.tsv \
    --folds_path /home/yangchenyu/Data_Imputation/imputation/results/wikituples/ablation_study/num_retrieved_tuples/folds.json \
    --output_path /home/yangchenyu/Data_Imputation/imputation/results/wikituples/ablation_study/num_retrieved_tuples/gpt4o-mini_wikituples_evidence_confidence_reranker_top40.jsonl \
    --top_k 40