python impute_wo_evidence.py \
    --api_url "https://vip.yi-zhan.top/v1/chat/completions" \
    --api_key "OPENAI_API_KEY" \
    --model "gpt-4o" \
    --temperature 0.3 \
    --data_dir "/home/yangchenyu/Data_Imputation/data/wikituples" \
    --input_file "/home/yangchenyu/Data_Imputation/data/wikituples/missing_tables.jsonl" \
    --output_path "/home/yangchenyu/Data_Imputation/imputation/results/wikituples/gpt4o_wikituples_wo_evidence.jsonl"
