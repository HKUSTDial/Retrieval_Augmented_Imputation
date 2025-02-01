import os
import argparse 
from collections import defaultdict


from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from model import SiameseRetriever
from utils import convert_file_to_tuple
import json


def calculate_recall(topk_pids, qrels, K):
    recall_sum = 0.0
    num_queries = len(topk_pids)

    for qid, retrieved_docs in topk_pids.items():
        retrieved_docs = set(retrieved_docs[:K])
        relevant_docs = set(qrels[qid])

        intersection = relevant_docs.intersection(retrieved_docs)
        recall = len(intersection) / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
        recall_sum += recall
 
    # 计算平均Recall Rate
    recall_rate = recall_sum / num_queries
    recall_rate = round(recall_rate, 3)
    print("Recall@{} =".format(K), recall_rate)
    return recall_rate
    

def calculate_success(topk_pids, qrels, K):
    success_at_k = []

    for qid, retrieved_docs in topk_pids.items():
        topK_docs = set(retrieved_docs[:K])
        relevant_docs = set(qrels[qid])

        if relevant_docs.intersection(topK_docs):
            success_at_k.append(1)

    success_at_k_avg = sum(success_at_k) / len(qrels)
    success_at_k_avg = round(success_at_k_avg, 3)
    
    print("Success@{} =".format(K), success_at_k_avg)
    return success_at_k_avg

 
def normalize_list(input_list):
    max_value = max(input_list)
    min_value = min(input_list)
    normalized_list = [(x - min_value) / (max_value - min_value) for x in input_list]
    return normalized_list

def eval_step(args, folder, version, docs, qrels, fold_name='dev'):
    retriever = SiameseRetriever.load(document_store=InMemoryDocumentStore(), load_dir=os.path.join(args.save_model_dir, folder), max_seq_len=256, default_path=args.default_path)
    index_name = args.model_name + '_' + version + '_' + folder + '_new_' + args.dataset_name
    index_path = os.path.join(args.temp_index_path, index_name) + '.faiss'
    print(index_path, index_name)
    
    if not os.path.exists(index_path):
        document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", index=index_name)

        document_store.write_documents(docs)

        document_store.update_embeddings(retriever)
        document_store.save(index_path=index_path)
    else:
        document_store = FAISSDocumentStore(faiss_index_path=index_path)

    rank_result = {}

    f_writer = open('./retrieval_results/Siamese_' + 'dataset_v4_new_' + args.dataset_name + '_top100_res_with_score.tsv', 'w')

    #with open(args.data_path + 'queries.' + fold_name + '.tsv', 'r') as f:
    with open(os.path.join(args.data_path, 'queries.tsv'), 'r') as f:
        #with open('/home/yangchenyu/Reranker/data/wikituples/queries.tsv', 'r') as f:
        for line in f:
            line = line.strip()
            qid, query = int(line[:line.index('\t')]), line[line.index('\t')+1:]
            if args.mask:
                query = query.replace('N/A', '<MASK>')

            top_documents = retriever.retrieve(query, top_k=100, document_store=document_store)
            document_ids, document_text, document_scores = [],[],[]
            for document in top_documents:
                document_ids.append(document.id)
                document_text.append(document.content)
                document_scores.append(document.score)
            
            # document_scores = normalize_list(document_scores)
                
            for rank, d_id in enumerate(document_ids):
                # rank_record = '\t'.join([str(qid), str(rank), str(d_id), str(document_scores[rank])])
                rank_record = '\t'.join([str(qid), str(d_id), str(document_scores[rank])])
                f_writer.write(rank_record + '\n')

            rank_result[qid] = [int(doc_id) for doc_id in document_ids]
    if fold_name == 'dev':
        recall = calculate_recall(rank_result, qrels, 100)
        return recall
    else:
        for K in [1, 5, 10, 20, 50, 100]:
            calculate_recall(rank_result, qrels, K)
            # calculate_precision(topK_pids, qrels, K)
            calculate_success(rank_result, qrels, K)
    



def main():
    parser = argparse.ArgumentParser(description="Tuple Learning for retrieval")
    
    # model setting
    parser.add_argument('--model_name', required=True, default='DPR', type=str, help='name of the model')
    parser.add_argument('--dataset_name', required=True, default='wikituples', type=str, help='name of the dataset')
    parser.add_argument('--best_model_file', required=False, type=str, default=None, help='path of best model')
    parser.add_argument('--save_model_dir', required=True, type=str, help='directory that saves the model')
    parser.add_argument('--final_model_dir', required=False, type=str, help='directory that saves the final model')
    parser.add_argument('--default_path', required=True, type=str, help='path of pre-trained model')
    parser.add_argument('--temp_index_path', required=True, type=str, help='path of temporary index')
    parser.add_argument('--data_path', required=True, type=str, help='path of data')
    parser.add_argument('--num_retrieved', default=100, type=int, help='number of retrieved tuples')
    parser.add_argument('--mask', default=False, type=bool, help='whether to replace N/A with <mask>')

    args = parser.parse_args()
    print(args.mask)
    
    test_qrels = defaultdict(list)
    version = args.save_model_dir.split('/')[-1]
    with open(os.path.join(args.data_path, 'new_qrels.tsv'), 'r') as f:
        for line in f:
            qid, docid, score = line.strip().split('\t')
            qid, docid = int(qid), int(docid)
            test_qrels[qid].append(docid)
    
    docs = convert_file_to_tuple(file_path=os.path.join(args.data_path, 'collection_2.tsv'))
    eval_step(args, args.best_model_file, version, docs, test_qrels, 'test')



    
if __name__ == "__main__":
    main()

