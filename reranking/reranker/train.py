# Copyright 2021 Reranker Author. All rights reserved.
# Code structure inspired by HuggingFace run_glue.py in the transformers library.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections import defaultdict

from reranker import Reranker, RerankerDC
from reranker import RerankerTrainer, RerankerDCTrainer
from reranker.data import GroupedTrainDataset, PredictionDataset, GroupCollator
from reranker.arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

logger = logging.getLogger(__name__)


def load_qrels(qrel_path):
    """Load query relevance judgments from file."""
    qrels = defaultdict(list)
    logger.info("Loading qrels from %s", qrel_path)
    
    with open(qrel_path) as f:
        for line in f:
            qid, pid, label = line.strip().split('\t')
            qid, pid = int(qid), int(pid)
            qrels[qid].append(pid)
    
    return qrels


def calculate_success(topk_pids, qrels, K):
    """Calculate Success@K metric for retrieval evaluation."""
    success_at_k = []
    count = 0
    
    for qid, retrieved_docs in topk_pids.items():
        topK_docs = set(retrieved_docs[:K])
        relevant_docs = set(qrels[qid])
        
        if len(relevant_docs) == 0:
            continue
        count += 1
        
        if relevant_docs.intersection(topK_docs):
            success_at_k.append(1)
    
    success_at_k_avg = sum(success_at_k) / count if count > 0 else 0.0
    success_at_k_avg = round(success_at_k_avg, 3)
    
    logger.info("Success@%d = %.3f", K, success_at_k_avg)
    return success_at_k_avg


def evaluate_predictions(rank_score_path, qrel_path, K_values=[1, 5, 10, 20, 50, 100]):
    """Evaluate predictions by calculating success metrics."""
    if not os.path.exists(rank_score_path):
        logger.warning("Score file %s does not exist, skipping evaluation", rank_score_path)
        return
    
    if not os.path.exists(qrel_path):
        logger.warning("Qrel file %s does not exist, skipping evaluation", qrel_path)
        return
    
    # Load qrels
    qrels = load_qrels(qrel_path)
    
    # Load predictions and group by query
    topk_pids = defaultdict(list)
    with open(rank_score_path, 'r') as f:
        for line in f:
            qid, pid, score = line.strip().split('\t')
            qid, pid, score = int(qid), int(pid), float(score)
            topk_pids[qid].append((pid, score))
    
    # Sort by score for each query
    for qid in topk_pids:
        topk_pids[qid].sort(key=lambda x: x[1], reverse=True)
        topk_pids[qid] = [pid for pid, _ in topk_pids[qid]]
    
    logger.info("Evaluating %d queries", len(topk_pids))
    
    # Calculate success metrics for different K values
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"{'K':<8} {'Success@K':<12}")
    print("-"*50)
    
    success_scores = []
    for K in K_values:
        success = calculate_success(topk_pids, qrels, K)
        success_scores.append(success)
        print(f"{K:<8} {success:<12.3f}")
    
    print("="*50)
    return success_scores


def main():
    # Parse arguments for model, data, and training configurations
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    # Check if output directory already exists and is not empty
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set random seed for reproducibility
    set_seed(training_args.seed)

    num_labels = 1

    # Load model configuration
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    # Select model class based on whether distance cache is used
    _model_class = RerankerDC if training_args.distance_cache else Reranker

    # Load the model
    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Load training dataset if training is enabled
    if training_args.do_train:
        train_dataset = GroupedTrainDataset(
            data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args
        )
    else:
        train_dataset = None


    # Initialize our Trainer
    _trainer_class = RerankerDCTrainer if training_args.distance_cache else RerankerTrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
    )

    # Training loop
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # Save tokenizer to the same directory as the model for easy deployment
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        trainer.evaluate()

    if training_args.do_predict:
        logging.info("*** Prediction ***")

        # Check if output score file already exists
        if os.path.exists(data_args.rank_score_path):
            if os.path.isfile(data_args.rank_score_path):
                raise FileExistsError(f'Score file {data_args.rank_score_path} already exists')
            else:
                raise ValueError(f'Should specify a file name')
        else:
            # Create output directory if it doesn't exist
            score_dir = os.path.split(data_args.rank_score_path)[0]
            if not os.path.exists(score_dir):
                logger.info(f'Creating score directory {score_dir}')
                os.makedirs(score_dir)

        # Load prediction dataset
        test_dataset = PredictionDataset(
            data_args.pred_path, tokenizer=tokenizer,
            max_len=data_args.max_len,
        )
        assert data_args.pred_id_file is not None

        # Read query IDs and passage IDs from the prediction ID file
        pred_qids = []
        pred_pids = []
        with open(data_args.pred_id_file) as f:
            for l in f:
                q, p = l.split()
                pred_qids.append(q)
                pred_pids.append(p)

        # Run prediction
        pred_scores = trainer.predict(test_dataset=test_dataset).predictions

        # Save scores to file
        if trainer.is_world_process_zero():
            assert len(pred_qids) == len(pred_scores)
            with open(data_args.rank_score_path, "w") as writer:
                for qid, pid, score in zip(pred_qids, pred_pids, pred_scores):
                    writer.write(f'{qid}\t{pid}\t{score}\n')
            
            # Evaluate predictions if qrel file is provided
            if data_args.qrel_path:
                logger.info("Evaluating predictions...")
                evaluate_predictions(data_args.rank_score_path, data_args.qrel_path)
            else:
                logger.info("No qrel file provided, skipping evaluation")



if __name__ == "__main__":
    main()
