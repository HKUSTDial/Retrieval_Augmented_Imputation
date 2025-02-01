CUDA_VISIBLE_DEVICES=6 python test.py \
  --dataset_name 'business' \
  --search_results_path '../data/business/triples.train.jsonl' \
  --search_results_path '../results/retrieval/business_top100_res_with_score.tsv' \
  --reranker_path '../experiment/business' \