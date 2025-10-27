import logging
import argparse 
from src.dense import SiameseRetriever
from haystack.utils import fetch_archive_from_http
from haystack.document_stores import InMemoryDocumentStore
import torch
import os
from typing import Optional, Any
from pathlib import Path
import json
import random

def _read_dpr_json(
    file: str,
    max_samples: Optional[int] = None,
    num_hard_negatives: int = 1,
    num_positives: int = 1,
    shuffle_negatives: bool = True,
    shuffle_positives: bool = False,
):
    
    if Path(file).suffix.lower() == ".jsonl":
        dicts = []
        with open(file, encoding="utf-8") as f:
            for line in f:
                dicts.append(json.loads(line))
    else:
        with open(file, encoding="utf-8") as f:
            dicts = json.load(f)

    if max_samples:
        dicts = random.sample(dicts, min(max_samples, len(dicts)))
    
    standard_dicts = []
    for dict in dicts:
        sample = {}
        sample["question"] = dict["question"]
        positives_ctxs = dict["positive_ctxs"]
        negative_ctxs = dict["hard_negative_ctxs"]
        if shuffle_positives:
            random.shuffle(positives_ctxs)
        if shuffle_negatives:
            random.shuffle(negative_ctxs)
        for passage in positives_ctxs[:num_positives]:
            sample['pos_doc'] = passage["text"]
            for passage in negative_ctxs[:num_hard_negatives]:
                sample['neg_doc'] = passage["text"]
                standard_dicts.append(sample)

    return standard_dicts


def main():
    parser = argparse.ArgumentParser(description="Tuple Learning for retrieval")
    
    # model setting
    parser.add_argument('--file_dir', required=True, type=str, help='file that contains necessary data')
    parser.add_argument('--train_filename', required=True, type=str, help='name of training file')
    parser.add_argument('--dev_filename', required=True, type=str, help='name of training file')
    parser.add_argument('--embedding_model', required=True, type=str, help='passage model')
    parser.add_argument('--max_seq_len', default=512, type=int, required=False, help='maximun length of query')
    parser.add_argument('--loss_function', default='dot_product', type=str, required=False, help='cos_sim, dot_product')

    # training setting
    parser.add_argument('--training_epochs', default=10, type=int, required=False, help='')
    parser.add_argument('--evaluate_every', default=1000, type=int, required=False, help='')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='')
    parser.add_argument('--num_positives', default=1, type=int, required=False, help='')
    parser.add_argument('--num_hard_negatives', default=3, type=int, required=False, help='')
    parser.add_argument('--checkpoint_every', default=1000, type=int, required=False, help='')
    parser.add_argument('--checkpoint_root_dir', default='model_checkpoints', type=str, required=False, help='')
    
    # output setting
    parser.add_argument('--save_dir', required=True, type=str, help='')

    args = parser.parse_args()

    args_dict = vars(args)
    for key, value in args_dict.items():
        print(f"{key}: {value}")

    # 检测gpu的数量
    if torch.cuda.is_available():
        # 获取GPU设备数量
        gpu_count = torch.cuda.device_count()
        print(f"Find {gpu_count} available GPU.")

    retriever = SiameseRetriever(
        document_store=InMemoryDocumentStore(),
        embedding_model=args.embedding_model,
        max_seq_len=args.max_seq_len,
    )
    
    retriever.train(
        data_dir=args.file_dir,
        train_filename=args.train_filename,
        dev_filename=args.dev_filename,
        test_filename=args.dev_filename,
        n_epochs=args.training_epochs,
        batch_size=args.batch_size,
        grad_acc_steps=4,
        save_dir=args.save_dir,
        evaluate_every=args.evaluate_every,
        checkpoint_every=args.checkpoint_every,
        checkpoints_to_keep=10,
        checkpoint_root_dir=args.checkpoint_root_dir,
        embed_title=False,
        num_positives=args.num_positives,
        num_hard_negatives=args.num_hard_negatives,
    )

    retriever.save(args.save_dir)

if __name__ == "__main__":
    main()