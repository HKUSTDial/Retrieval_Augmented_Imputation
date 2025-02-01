from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
import argparse
import jsonlines
from collections import defaultdict


def calculate_recall(topk_pids, qrels, K):
    recall_sum = 0.0
    num_queries = len(qrels)

    for qid, qrel in qrels.items():
        if qid not in topk_pids:
            continue
        retrieved_docs = set(topk_pids[qid][:K])
        relevant_docs = set(qrel)

        intersection = relevant_docs.intersection(retrieved_docs)
        recall = len(intersection) / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
        recall_sum += recall

    # 计算平均Recall Rate
    recall_rate = recall_sum / num_queries
    print("Recall@{} =".format(K), recall_rate)
    

def calculate_success(topk_pids, qrels, K):
    success_at_k = []
    total = len(qrels)
    for qid, qrel in qrels.items():
        if qid not in topk_pids:
            continue
        relevant_docs = set(qrel)
        topK_docs = set(topk_pids[qid][:K]) if qid in topk_pids else set()
        if relevant_docs.intersection(topK_docs):
            success_at_k.append(1)
            
    success_at_k_avg = sum(success_at_k) / total
    success_at_k_avg = round(success_at_k_avg, 3)
    
    print("Success@{} =".format(K), success_at_k_avg)



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
            # qid, _, pid, label = line.strip().split('\t')
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

    search_results = load_search_res(args.search_results_path)

    reranker = MonoT5(pretrained_model_name_or_path=args.reranker_path)

    rerank_results = defaultdict(list)

    for qid, query in queries.items():
        tables = []
        for docid in search_results[qid]:
            if docid in qrels[qid]:
                tables.append(Text(collection[docid], {'docid': docid}, 1))
            else:
                tables.append(Text(collection[docid], {'docid': docid}, 0))
        
        query = Query(query.strip())
        reranked = reranker.rerank(query, tables)
        
        fout = open('../results/rerank/' + args.dataset_name + '.test.tsv', 'a')
        for i in range(0, 100):
            rerank_results[qid].append(reranked[i].metadata['docid'])
            fout.write(str(qid) + '\t' + str(reranked[i].metadata['docid']) + '\t' + str(i) + '\t' + str(reranked[i].score) + '\n')
        fout.close()
        
    for K in [1, 5, 10, 20, 50, 100]:
        calculate_recall(rerank_results, qrels, K)
        calculate_success(rerank_results, qrels, K)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reranking and Evaluation Script")
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of dataset')
    # parser.add_argument('--query_path', type=str, required=True, help='Path to the query file')
    # parser.add_argument('--collection_path', type=str, required=True, help='Path to the collection file')
    # parser.add_argument('--qrels_path', type=str, required=True, help='Path to the qrels file')
    parser.add_argument('--search_results_path', type=str, required=True, help='Path to the search results file')
    parser.add_argument('--reranker_path', type=str, required=True, help='Path to the reranker model')

    args = parser.parse_args()
    main(args)

