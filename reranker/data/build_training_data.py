import jsonlines
import argparse
from collections import defaultdict
import random
import json


def load_queries(path):
    print("Loading queries...")

    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            qid, query = line[:line.index('\t')], line[line.index('\t')+1:]
            qid = int(qid)
            queries[qid] = query

    return queries

def load_collection(path):
    print("Loading collection...")
    
    collection = {}
    with open(path) as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            pid, passage = line[:line.index('\t')], line[line.index('\t')+1:]
            collection[int(pid)] = passage

    return collection

def load_qrels(qrel_path):
    qrels = defaultdict(list)
    print("Loading the qrels from", qrel_path, "...")

    with open(qrel_path) as f:
        for line in f:
            qid, pid, label = line.strip().split('\t')
            qid, pid = int(qid), int(pid)
            qrels[qid].append(pid)

    return qrels

def load_search_res(res_path):
    results = defaultdict(list)
    print("Loading the search result from", res_path, "...")

    with open(res_path) as f:
        for line in f:
            qid, docid, _ = line.strip().split('\t')
            results[int(qid)].append(int(docid))
    return results


def main(args):
    query_path = '../data/' + args.dataset_name + '/queries.test.tsv'
    collection_path = '../data/' + args.dataset_name + '/collection.tsv'
    qrels_path = '../data/' + args.dataset_name + '/qrels.test.tsv'

    queries = load_queries(query_path)
    collection = load_collection(collection_path)
    qrels = load_qrels(qrels_path)
    search_results = load_search_res(args.retrieval_file)

    fout = jsonlines.open(args.dataset_name + '/triples.train.jsonl', 'w')
    for qid, query in queries.items():
        positives = qrels[qid]
        item = {'query': query, 'positive': [collection[docid] for docid in positives]}
        retrieved_docs = search_results[qid]
        retrieved_docs = [docid for docid in retrieved_docs if docid not in positives]
        item['negative'] = [collection[docid] for docid in random.sample(retrieved_docs[:20], 11)]
        fout.write(item)
    fout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preparation for Ranking")
    parser.add_argument('--dataset_name', type=str, required=True, help='name of dataset')
    parser.add_argument('--retrieval_file', type=str, required=True, help='Path to the search results file')
    
    args = parser.parse_args()
    main(args)

