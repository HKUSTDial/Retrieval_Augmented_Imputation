
from typing import Callable, Dict, List, Optional
from haystack.schema import Document


def calculate_recall(topk_pids, qrels, K):
    recall_sum = 0.0
    num_queries = len(topk_pids)

    for qid, retrieved_docs in topk_pids.items():
        retrieved_docs = set(retrieved_docs[:K])
        relevant_docs = set(qrels[qid])

        intersection = relevant_docs.intersection(retrieved_docs)
        recall = len(intersection) / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
        recall_sum += recall

    # Calculate average Recall Rate
    recall_rate = recall_sum / num_queries
    print("Recall@{} =".format(K), recall_rate)
    return recall_rate
    

def calculate_success(topk_pids, qrels, K):
    success_at_k = []

    for qid, retrieved_docs in topk_pids.items():
        topK_docs = set(retrieved_docs[:K])
        relevant_docs = set(qrels[qid])

        if relevant_docs.intersection(topK_docs):
            success_at_k.append(1)

    success_at_k_avg = sum(success_at_k) / len(topk_pids)
    success_at_k_avg = round(success_at_k_avg, 3)
    
    print("Success@{} =".format(K), success_at_k_avg)
    return success_at_k_avg


def normalize_list(input_list):
    print(input_list)
    max_value = max(input_list)
    min_value = min(input_list)
    normalized_list = [(x - min_value) / (max_value - min_value) for x in input_list]
    return normalized_list



def convert_file_to_tuple(file_path: str,
    add_title:Optional[bool] = False,
    clean_func: Optional[Callable] = None,
    split_paragraphs: bool = False,
    encoding: Optional[str] = None,
    id_hash_keys: Optional[List[str]] = None,
) -> List[Document]:

    documents = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            t_id, tuple_text = line[:line.index('\t')], line[line.index('\t')+1:]
            documents.append(Document(id=t_id, content=tuple_text, content_type='text'))
    return documents