MODEL="gpt-4o"
API_URL="your_api_url_here"
API_KEY="your_api_key_here"

BASE_PATH="/path/to/your/data"
DATASET="restaurants"
TOP_K=5
THRESHOLD=0.9

# Retrieval results file path
RETRIEVAL_RESULTS="/path/to/your/retrieval/results.txt"

python impute.py \
    --model ${MODEL} \
    --api_url ${API_URL} \
    --api_key ${API_KEY} \
    --data_path ${DATA_DIR} \
    --retrieval_results_path "${RETRIEVAL_RESULTS}" \
    --output_path "${OUTPUT_DIR}/${MODEL}_imputation_results_top_${TOP_K}_threshold_${THRESHOLD}.jsonl" \
    --dataset ${DATASET} \
    --missing_columns "address" \
    --threshold ${THRESHOLD} \
    --top_k ${TOP_K}