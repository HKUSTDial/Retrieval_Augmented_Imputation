CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model_name='siamese' \
  --dataset_name='show_movie' \
  --num_retrieved=100 \
  --save_model_dir='/data/yangchenyu/Data_Imputation_whole/retriever/model_checkpoints_siamese/dataset_v4/epoch_1_step_10000' \
  --default_path='/home/yangchenyu/pre-trained-models/bert-base-uncased' \
  --temp_index_path='/home/yangchenyu/Retrieval_Augmented_Imputation/retriever/index' \
  --data_path='/home/yangchenyu/Data_Imputation/data/show_movie/annotated_data' 