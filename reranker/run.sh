CUDA_VISIBLE_DEVICES=6 python train.py \
  --base_model 'castorini/monot5-base-msmarco-10k' \
  --triples_path '../data/business/triples.train.jsonl' \
  --output_model_path '../experiment/business' \
  --save_every_n_steps 10000 \
  --epochs 30 \
