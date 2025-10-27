import os
import argparse 
from collections import defaultdict


from haystack.document_stores import FAISSDocumentStore, InMemoryDocumentStore
from src.dense import SiameseRetriever
from utils import convert_file_to_tuple
import json


def calculate_recall(topk_pids, qrels, K, verbose=True):
    """
    Calculate Recall@K metric for retrieval evaluation.
    
    Args:
        topk_pids: Dictionary mapping query IDs to retrieved document IDs
        qrels: Dictionary mapping query IDs to relevant document IDs
        K: Number of top documents to consider
        verbose: Whether to print the result
    
    Returns:
        Average recall rate across all queries
    """
    recall_sum = 0.0
    num_queries = len(topk_pids)

    for qid, retrieved_docs in topk_pids.items():
        retrieved_docs = set(retrieved_docs[:K])
        relevant_docs = set(qrels[qid])

        intersection = relevant_docs.intersection(retrieved_docs)
        recall = len(intersection) / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
        recall_sum += recall
 
    # Calculate average recall rate
    recall_rate = recall_sum / num_queries
    recall_rate = round(recall_rate, 3)
    if verbose:
        print(f"Recall@{K:2d} = {recall_rate:.3f}")
    return recall_rate
    

def calculate_success(topk_pids, qrels, K, verbose=True):
    """
    Calculate Success@K metric for retrieval evaluation.
    
    Args:
        topk_pids: Dictionary mapping query IDs to retrieved document IDs
        qrels: Dictionary mapping query IDs to relevant document IDs
        K: Number of top documents to consider
        verbose: Whether to print the result
    
    Returns:
        Average success rate across all queries
    """
    success_at_k = []

    for qid, retrieved_docs in topk_pids.items():
        topK_docs = set(retrieved_docs[:K])
        relevant_docs = set(qrels[qid])

        # Check if any relevant document is in top K
        if relevant_docs.intersection(topK_docs):
            success_at_k.append(1)

    success_at_k_avg = sum(success_at_k) / len(qrels)
    success_at_k_avg = round(success_at_k_avg, 3)
    
    if verbose:
        print(f"Success@{K:2d} = {success_at_k_avg:.3f}")
    return success_at_k_avg

 
def evaluation(args, docs, qrels, fold_name='dev'):
    # Load the Siamese retriever model directly from save_model_dir
    retriever = SiameseRetriever.load(
        document_store=InMemoryDocumentStore(), 
        load_dir=args.save_model_dir, 
        max_seq_len=256, 
        default_path=args.default_path
    )
    
    # Create index name and path
    index_name = f"{args.model_name}_{args.dataset_name}"
    index_path = os.path.join(args.temp_index_path, index_name) + '.faiss'
    print(f"Index path: {index_path}")
    print(f"Index name: {index_name}")
    
    # Ensure the temp_index_path directory exists
    os.makedirs(args.temp_index_path, exist_ok=True)
    
    # Create or load FAISS document store
    if not os.path.exists(index_path):
        print("Creating new FAISS index...")
        document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", index=index_name)
        document_store.write_documents(docs)
        document_store.update_embeddings(retriever)
        document_store.save(index_path=index_path)
    else:
        print("Loading existing FAISS index...")
        document_store = FAISSDocumentStore(faiss_index_path=index_path)

    # Initialize result storage
    rank_result = {}
    
    # Create output directory if it doesn't exist
    os.makedirs('./retrieval_results', exist_ok=True)
    
    # Open output file for writing results
    output_file = f'./retrieval_results/Siamese_{args.dataset_name}_top100_res_with_score.tsv'
    f_writer = open(output_file, 'w')

    # Process queries and retrieve documents
    print("Processing queries...")
    with open(os.path.join(args.data_path, 'queries.tsv'), 'r') as f:
        for line in f:
            line = line.strip()
            qid, query = int(line[:line.index('\t')]), line[line.index('\t')+1:]
            
            # Apply mask if specified
            if args.mask:
                query = query.replace('N/A', '<MASK>')

            # Retrieve top documents
            top_documents = retriever.retrieve(query, top_k=100, document_store=document_store)
            document_ids, document_text, document_scores = [], [], []
            
            for document in top_documents:
                document_ids.append(document.id)
                document_text.append(document.content)
                document_scores.append(document.score)
                
            # Write results to file
            for rank, d_id in enumerate(document_ids):
                rank_record = '\t'.join([str(qid), str(d_id), str(document_scores[rank])])
                f_writer.write(rank_record + '\n')

            rank_result[qid] = [int(doc_id) for doc_id in document_ids]
    # Close the output file
    f_writer.close()
    
    # Calculate evaluation metrics
    if fold_name == 'dev':
        recall = calculate_recall(rank_result, qrels, 100)
        return recall
    else:
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"{'Metric':<12} {'K=1':<8} {'K=5':<8} {'K=10':<8} {'K=20':<8} {'K=50':<8} {'K=100':<8}")
        print("-"*50)
        
        # Calculate metrics for different K values
        recall_scores = []
        success_scores = []
        
        for K in [1, 5, 10, 20, 50, 100]:
            recall = calculate_recall(rank_result, qrels, K, verbose=False)
            success = calculate_success(rank_result, qrels, K, verbose=False)
            recall_scores.append(recall)
            success_scores.append(success)
        
        # Print formatted results
        print(f"{'Recall@K':<12} {recall_scores[0]:<8.3f} {recall_scores[1]:<8.3f} {recall_scores[2]:<8.3f} {recall_scores[3]:<8.3f} {recall_scores[4]:<8.3f} {recall_scores[5]:<8.3f}")
        print(f"{'Success@K':<12} {success_scores[0]:<8.3f} {success_scores[1]:<8.3f} {success_scores[2]:<8.3f} {success_scores[3]:<8.3f} {success_scores[4]:<8.3f} {success_scores[5]:<8.3f}")
        print("="*50)
    



def main():
    parser = argparse.ArgumentParser(description="Siamese Retrieval Model Evaluation")
    
    # Model and dataset settings
    parser.add_argument('--model_name', required=True, type=str, help='Name of the model')
    parser.add_argument('--dataset_name', required=True, type=str, help='Name of the dataset')
    parser.add_argument('--save_model_dir', required=True, type=str, help='Directory containing the saved model')
    parser.add_argument('--default_path', required=True, type=str, help='Path to pre-trained model')
    parser.add_argument('--temp_index_path', required=True, type=str, help='Path for temporary index storage')
    parser.add_argument('--data_path', required=True, type=str, help='Path to dataset')
    parser.add_argument('--num_retrieved', default=100, type=int, help='Number of retrieved documents')
    parser.add_argument('--mask', default=False, type=bool, help='Whether to replace N/A with <MASK>')

    args = parser.parse_args()
    print(f"Mask setting: {args.mask}")
    
    # Load query relevance judgments
    print("Loading query relevance judgments...")
    test_qrels = defaultdict(list)
    with open(os.path.join(args.data_path, 'qrels.tsv'), 'r') as f:
        for line in f:
            qid, docid, score = line.strip().split('\t')
            qid, docid = int(qid), int(docid)
            test_qrels[qid].append(docid)
    
    # Load document collection
    print("Loading document collection...")
    docs = convert_file_to_tuple(file_path=os.path.join(args.data_path, 'collection.tsv'))
    
    # Run evaluation
    print("Starting evaluation...")
    evaluation(args, docs, test_qrels, 'test')



    
if __name__ == "__main__":
    main()

