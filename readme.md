# Data Imputation with Limited Data Redundancy Using Data Lakes
## Introduction
In this paper, we explore the potential of leveraging large language models (LLMs) for the imputation of massive small tables. Despite the promise of LLMs in capturing external knowledge, their application is hindered by issues of hallucination and lack of interpretability. To address these limitations, we propose a novel approach based on retrieval-augmented imputation RAI, utilizing tuple embeddings instead of traditional table embeddings. 

This repository contains the code for the paper **Data Imputation with Limited Data Redundancy Using Data Lakes**


## Data & Model
To facilitate research in retrieval-augmented missing value imputation, we release the first large-scale benchmark, **mvBench**, containing 15,143 incomplete tuples and 4.23 million tuples from the data lake. Detailed descriptions and analyses of these datasets are provided in our paper. Additionally, we release pretraining data we constructed for further research. The datasets and pretrained retriever model's checkpoints can be accessed through the following OSF repository link:
- [Access mvBench and Pretraining Materials](https://osf.io/kh2ps/?view_only=aade7da94ad04bf9887a2b631bb57a32)

This repository contains:
- **Five Datasets in mvBench**
- **Pretraining Data for our Retriever**
- **Checkpoints of our Pretrained Retriever**

### mvBench
The mvBench dataset is constructed from three main data sources:

 1. WikiTables-TURL: This dataset is sourced from "TURL: Table Understanding through Representation Learning" by Deng et al. (2022). The original dataset is available in [its GitHub repository](https://github.com/sunlab-osu/TURL/tree/release_ongoing) under the Apache License 2.0.
 2. Show Movie and Cricket Players: These datasets are obtained from "RetClean: Language-Model Based Data Cleaning via Retrieval" by Ahmad et al. (2023). The original datasets are available in [its GitHub repository ](https://github.com/qcri/RetClean).
 3. Education and Business: These datasets are collected from the [Chicago Open Data Portal](https://data.cityofchicago.org/).

After colleted the data, we process and anotate them to construct mvBench. Each dataset for Data Imputation in mvBench includes the following files: **queries.tsv, qrels.tsv, collection.tsv, folds.json**

 - **queries.tsv**: Lists tuples with missing values denoted by "N/A", each identified by a unique ID.
 - **qrels.tsv**: Contains query IDs, associated target tuple IDs, and their relevance scores.
 - **collection.tsv**: Each row contains a tuple ID and its corresponding complete tuple text.
 - **folds.json**: Specifies query IDs included in the train set and the test set, respectively.

### Model

For ease of use, we also provide our pretrained retriever model. You can find the checkpoint in the OSF repository for immediate use.


## Running Code
### Retrieval
Execute the following steps to train the retriever on your dataset:
1. Navigate to the retriever directory and modify the `train_siamese.sh` script, setting `file_dir` to your pretraining data location.
2. Run the script using the command:
   ```powershell
   ./train_siamese.sh
   ```

After training, use the `test_siamese.sh` script to build indexes for tuples and retrieve the top-k tuples. Adjust num_retrieved and the dataset for indexing as required, then execute:
```powershell
./test_siamese.sh
```

### Rerank
The reranker module leverages Pygaggle, a gaggle of deep neural architectures for text ranking and question answering. Follow these steps:

1. In the reranker directory, clone the Pygaggle repository and set up the required environment:
```powershell
git clone https://github.com/castorini/pygaggle.git 
```

2. Transfer `run.sh`, `test.sh`, `train.py`, and `test.py` from the reranker directory to Pygaggle.

3. Prepare training data for the reranker by running build_training_data.py in the ./reranker/data directory. Specify the dataset and retrieval results file, for example:
```powershell
python build_training_data.py  --dataset_name 'wikituples'  --retrieval_file '../../results/retrieval/wikituples_retrieval_results.tsv'
```

4. Fine-tune the reranker using:
```powershell
./run.sh
```

5. Rerank the retrieval results:
```powershell
./test.sh
```
### Data Imputation

The Imputation directory contains Jupyter notebooks (*.ipynb*) with code and evaluation for data imputation, both with and without the use of retrieved tuples.

## Installation 
```powershell
pip install haystack-ai
```
