
# Data Imputation with Limited Data Redundancy Using Data Lakes

## Introduction
Data imputation is critical for many data science applications. Traditional methods, leveraging statistics, integrity constraints, or machine learning, depend on sufficient within-table data redundancy to infer missing values. However, real-world datasets often lack such redundancy, necessitating external data sources. To address this challenge, we introduce **LakeFill**, a novel retrieval-augmented imputation framework that integrates large language models (LLMs) and data lakes.

This repository contains the implementation for the paper:
**Data Imputation with Limited Data Redundancy Using Data Lakes**

## Data & Model
To advance research in retrieval-augmented imputation, we introduce **mvBench**, the first large-scale benchmark in this domain. mvBench consists of **7,784 incomplete tuples** and **2.8 million tuples** sourced from a data lake. Detailed dataset descriptions and analyses are available in our paper. We also release our pretraining dataset and retriever model checkpoints for further research.

Access the datasets and pretrained retriever model here:
- [**mvBench and Pretraining Materials**](https://hkustgz-my.sharepoint.com/:f:/g/personal/cyang662_connect_hkust-gz_edu_cn/En4KpS23e6RGmYNAO_h-MVEBPA-tKEHkVzs8dZzxmu5iMw?e=PdZpm9)

### Contents of this Repository
- **mvBench Datasets**
- **Pretraining Data for the Retriever**
- **Checkpoints of the Pretrained Retriever**

### mvBench Dataset
mvBench is constructed from the following data sources:
1. **WikiTables-TURL** (Deng et al., 2022): Available in [this GitHub repository](https://github.com/sunlab-osu/TURL/tree/release_ongoing) under Apache License 2.0.
2. **Show Movie and Cricket Players** (Ahmad et al., 2023): Available in [this GitHub repository](https://github.com/qcri/RetClean).
3. **Education Data**: Collected from the [Chicago Open Data Portal](https://data.cityofchicago.org/).

After collection, the data is processed and annotated to form mvBench. Each dataset includes:
- **queries.tsv**: Tuples with missing values (marked as "N/A"), each assigned a unique ID.
- **qrels.tsv**: Query IDs mapped to target tuple IDs with relevance scores.
- **collection.tsv**: Complete tuples with their respective IDs.
- **folds.json**: Specifies train and test query splits.
- **answers.jsonl**: Contains ground truth imputation values.

### Pretrained Retriever Model
We provide a pretrained retriever for ease of use. The checkpoint is included in the dataset link above.

## Environment Setup
To set up the environment, you can install the required dependencies by running the following command:
```bash
pip install -r requirements.txt
```

## Running Code
### Retrieval
We train a retriever using contrastive learning and synthetic training data to encode tuples into embeddings. These embeddings are indexed in a vector database, enabling efficient similarity search across heterogeneous data lakes.

To train the retriever on your dataset, run the [`train_siamese.sh`](./retriever/train_siamese.sh) script:

```bash
#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=5,6,7 python train_siamese.py \
  --file_dir='pretraining_data' \
  --train_filename='train.json' \
  --dev_filename='dev.json' \
  --embedding_model='bert-base-uncased' \
  --save_dir='checkpoints' \
  --training_epochs=4 \
  --batch_size=16 \
  --max_seq_len=256 \
  --num_positives=1 \
  --num_hard_negatives=7 \
  --evaluate_every=10000000 \
  --checkpoint_every=10000 \
  --checkpoint_root_dir='checkpoints'
```

**Key Parameters:**
- `--file_dir`: Directory containing pretraining data.
- `--train_filename`: Training dataset file.
- `--dev_filename`: Validation dataset file.
- `--embedding_model`: Pretrained embedding model.
- `--num_positives`: Number of positive samples per instance.
- `--num_hard_negatives`: Number of hard negative samples per instance.

After training, use [`test_siamese.sh`](./retriever/test_siamese.sh) to build tuple indexes and retrieve top-k results.

### Reranking
The reranker refines retrieved tuples using a fine-grained comparison, ensuring that the top-k candidates (\( k \ll K \)) are the most relevant. We employ a **checklist-based approach** to construct high-quality training data for token-level reranking, mitigating labeling errors through carefully designed training groups.

#### Steps:
1. Set up the environment following instructions in [reranker README](reranking/reranker/readme.md).
2. Construct training groups using [`construct_training_groups.py`](reranking/annotation/construct_training_groups.py).
3. Train the reranker:

```bash
#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=3 python train.py \
    --output_dir ./checkpoints/cricket_players \
    --model_name_or_path ./pre-trained-models/Luyu/bert-base-mdoc-bm25 \
    --train_path ./data/cricket_players/generated_data.train.jsonl \
    --max_len 512 \
    --per_device_train_batch_size 1 \
    --train_group_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --num_train_epochs 30 \
    --save_steps 2000 \
    --seed 42 \
    --do_train \
    --logging_steps 20 \
    --overwrite_output_dir
```

**Key Parameters:**
- `--output_dir`: Directory to save trained model and logs.
- `--model_name_or_path`: Pretrained model for initialization.
- `--train_group_size`: Number of negatives per positive sample.
- `--num_train_epochs`: Total training epochs.

4. Test the reranker using `test.sh`.

### Data Imputation
We propose a **two-stage confidence-aware imputation approach** that ensures accurate and context-aware missing value imputation. The **Imputation** directory contains dataset-specific imputation implementations.

---
For further details, refer to our paper or contact us for additional support.
