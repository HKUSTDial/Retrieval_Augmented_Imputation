import json
import re
import argparse
import os
from collections import defaultdict
import openai

def generate_response(messages, api_key, api_base, model):
    """Generate response using OpenAI API"""
    openai.api_key = api_key
    openai.api_base = api_base
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=500
    )
    return response.choices[0].message.content

# New prompt template for evaluation
prompt_template = '''You are tasked with determining whether a candidate tuple can be used to fill in at least one of the missing values of a query tuple denoted as 'N/A'. The assessment should be made based on three dimensions: Existence, Relevance, and Compatibility.

### Dimensions:
1. Existence:
* Response Options: Yes, No
* Explanation: Check if the candidate tuple contains at least one of the missing attribute(s) denoted as 'N/A' needed to be filled in the query tuple, considering that attribute names may be different but represent the same concept. 

2. Relevance:
* Response Options: Highly Relevant, Somewhat Relevant, Not Relevant
* Explanation: Determine the degree to which the candidate tuple relates to the query tuple in terms of describing the same or related entities.

3. Logical Consistency:
* Response Options: Fully Consistent, Partially Consistent, Not Consistent
* Explanation: Evaluates whether the data from the candidate tuple logically aligns with the query tuple, pay special attention to factors such as temporal consistency, functional dependencies.

### Example 1:
Query: yukon legislative assembly current members attribute name value Scott_Kent attribute party value N/A attribute riding value N/A 

Candidate Tuple: 33rd yukon legislative assembly current members attribute name value Scott Kent attribute party value Yukon Party attribute riding value Riverdale North

Evaluation:
1. Existence: The candidate tuple provides the missing 'party' and 'riding' attributes for Scott Kent.
[Response] Yes
2. Relevance: The query seeks current members of the Yukon Legislative Assembly, and the candidate tuple refers to the 33rd assembly's current members. Assuming the 33rd assembly is current, it is highly relevant.
[Response] Highly Relevant
3. Logical Consistency: The candidate tuple states Scott Kent's party as Yukon Party and riding as Riverdale North. Verification is needed to confirm Riverdale North is a valid Yukon riding and Scott Kent's party affiliation. The name difference (Scott_Kent vs. Scott Kent) appears to be a formatting issue.
[Response] Fully Consistent

### Example 2:
Query: list of french films of 2005 attribute title value A_Very_Long_Engagement attribute director value Jean_Pierre_Jeunet attribute cast value N/A attribute genre value Romantic War attribute notes value Nominated for 2 Oscars. Another 16 wins and 21 nominations

Candidate Tuple: andré dussollier selected filmography attribute year value 2004 attribute title value A Very Long Engagement attribute role value Pierre-Marie Rouvières attribute director value Jean-Pierre Jeunet

Evaluation:
1. Existence: The candidate tuple does not contain the 'cast' attribute.
[Response] No
2. Relevance: The candidate tuple pertains to André Dussollier's role in a film, while the query seeks the cast of a different year's film.
[Response] Not Relevant
3. Logical Consistency: The years do not match, and the candidate tuple lacks the 'cast' attribute.
[Response] Not Consistent

Query: {query_tuple}
Candidate Tuple: {candidate_tuple}

Please provide your evaluation following the format in the examples, with [Response] tags for each dimension.'''



class DataAnnotator:
    def __init__(self, config):
        """
        Initialize the annotator with configuration
        
        Args:
            config: dict containing:
                - data_paths: dict with paths to required data files
                - openai_config: dict with OpenAI API settings
                - annotation_config: dict with annotation requirements
                - output_path: str, path to save results
        """
        self.config = config
        
        # Set up OpenAI
        openai.api_key = config['openai_config']['api_key']
        openai.api_base = config['openai_config']['api_base']
        self.gpt_model = config['openai_config']['model']
        
        # Set up annotation requirements
        self.max_annotations = config['annotation_config']['max_annotations_per_query']
        self.min_negative = config['annotation_config']['min_negative_samples']
        self.min_positive = config['annotation_config']['min_positive_samples']
        
        # Load data
        self.tuple_record = self._load_tuple_records()
        self.qrel_record = self._load_qrel_records()
        self.train_qids = self._load_train_qids()
        self.query_record = self._load_query_records()
        self.retrieved_results = self._load_retrieved_results()
        
        # Initialize query status
        self.query_status = defaultdict(lambda: {
            'annotation_count': 0,
            'positive_samples': 0,
            'negative_samples': 0,
            'processed_candidates': set()
        })
        
        # Load existing annotations if any
        self._load_existing_annotations()
    

    def extract_responses(self, text):
        """Extract dimension scores from GPT response"""
        responses = {
            'existence': None,
            'relevance': None,
            'consistency': None
        }
        
        # Use regex to extract content from [Response] tags
        response_pattern = r'\[Response\]\s*(.*?)(?=\n|$)'
        matches = re.finditer(response_pattern, text)
        
        responses_list = [match.group(1).strip() for match in matches]
        
        if len(responses_list) >= 3:
            responses['existence'] = responses_list[0]
            responses['relevance'] = responses_list[1]
            responses['consistency'] = responses_list[2]
        
        return responses



    def _load_tuple_records(self):
        records = {}
        with open(self.config['data_paths']['collection'], 'r') as f:
            for line in f:
                tid, tuple_text = line[:line.index('\t')], line[line.index('\t')+1:].strip()
                records[tid] = tuple_text
        return records
    

    def _evaluate_tuple_dimensions(self, query, candidate):
        prompt = prompt_template.format(
            query_tuple=query,
            candidate_tuple=candidate
        )
        messages = [
            {"role": "system", "content": "You are a data imputation expert."},
            {"role": "user", "content": prompt}
        ]
        
        response = generate_response(messages, self.config['openai_config']['api_key'], 
                                   self.config['openai_config']['api_base'], 
                                   self.config['openai_config']['model'])
        dimension_scores = self.extract_responses(response)
        
        return {
            "full_response": response,
            "dimension_scores": dimension_scores
        }



    def _load_qrel_records(self):
        qrel_record = defaultdict(list)
        with open(self.config['data_paths']['qrels'], 'r') as f:
            for line in f:
                qid, tid, _ = line.strip().split('\t')
                qrel_record[int(qid)].append(tid)
        return qrel_record

    def _load_train_qids(self):
        with open(self.config['data_paths']['folds'], 'r') as f:
            folds = json.load(f)
            return folds['train']

    def _load_query_records(self):
        records = {}
        with open(self.config['data_paths']['queries'], 'r') as f:
            for line in f:
                tid, tuple_text = line[:line.index('\t')], line[line.index('\t')+1:].strip()
                if int(tid) in self.train_qids: 
                    records[int(tid)] = tuple_text
        return records

    def _load_retrieved_results(self):
        retrieved_results = defaultdict(list)
        with open(self.config['data_paths']['retrieved_results'], 'r') as f:
            for line in f:
                qid, tid, _ = line.strip().split('\t')
                retrieved_results[int(qid)].append(tid)
        return retrieved_results

    def _load_existing_annotations(self):
        try:
            with open(self.config['output_path'], 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    qid = record['query_id']
                    candidate_id = record['candidate_id']
                    
                    # Update query status
                    self.query_status[qid]['annotation_count'] += 1
                    self.query_status[qid]['processed_candidates'].add(candidate_id)  # Record processed candidate
                    
                    # Calculate score and determine if it's positive or negative sample
                    score = self.calculate_score(record['evaluation']['dimension_scores'])
                    if score is not None and score >= 2:
                        self.query_status[qid]['positive_samples'] += 1
                    else:
                        self.query_status[qid]['negative_samples'] += 1
        except FileNotFoundError:
            print("[INFO] No existing annotation results found. Starting fresh annotation session.")

    def calculate_score(self, dimension_scores):
        """Calculate the score for a candidate sample"""
        if 'no' in dimension_scores['existence'].lower():
            return None
        
        relevance_score = 0
        consistency_score = 0
        
        relevance_lower = dimension_scores['relevance'].lower()
        consistency_lower = dimension_scores['consistency'].lower()
        
        if 'highly relevant' in relevance_lower:
            relevance_score = 2
        elif 'somewhat relevant' in relevance_lower:
            relevance_score = 1
            
        if 'fully consistent' in consistency_lower:
            consistency_score = 2
        elif 'partially consistent' in consistency_lower:
            consistency_score = 1
        
        return relevance_score + consistency_score

    def should_stop_query(self, qid):
        """Check if we should stop processing this query"""
        status = self.query_status[qid]
        return (status['annotation_count'] >= self.max_annotations or
                (status['positive_samples'] >= self.min_positive and 
                 status['negative_samples'] >= self.min_negative))

    def annotate(self):
        """Main annotation process"""
        print("\n" + "="*80)
        print("ANNOTATION SESSION STARTED")
        print("="*80 + "\n")
        
        for qid in self.train_qids:
            if self.should_stop_query(qid):
                print(f"[SKIP] Query {qid}: Already completed (satisfies stopping criteria)")
                continue

            print(f"\n{'─'*80}")
            print(f"[PROCESSING] Query {qid}")
            print(f"{'─'*80}")
            print(f"Query Text: {self.query_record[qid]}")
            print(f"Status: {self.query_status[qid]['annotation_count']} annotations | "
                  f"{self.query_status[qid]['positive_samples']} positive | "
                  f"{self.query_status[qid]['negative_samples']} negative")
            
            for candidate_id in self.retrieved_results[qid]:
                if self.should_stop_query(qid):
                    break
                    
                if candidate_id in self.query_status[qid]['processed_candidates']:
                    continue
                
                self._process_candidate(qid, candidate_id)
                
        self._print_final_statistics()
        
        # Generate training data after annotation completion
        self.generate_training_data()

    def _process_candidate(self, qid, candidate_id):
        """Process a single candidate for a query"""
        candidate_tuple = self.tuple_record.get(candidate_id, "")
        if not candidate_tuple:
            return

        result = self._evaluate_tuple_dimensions(
            self.query_record[qid], 
            candidate_tuple
        )
        
        score = self.calculate_score(result['dimension_scores'])
        self._update_status_and_save(qid, candidate_id, result, score)

    def _update_status_and_save(self, qid, candidate_id, result, score):
        """Update query status and save annotation result"""
        # Update status
        self.query_status[qid]['annotation_count'] += 1
        self.query_status[qid]['processed_candidates'].add(candidate_id)
        
        if score is not None and score >= 2:
            self.query_status[qid]['positive_samples'] += 1
        else:
            self.query_status[qid]['negative_samples'] += 1
        
        print('--------------------------------')
        print(f"Query {qid} candidate {candidate_id} score: {score}")
        print(result)

        # Save annotation result
        annotation_record = {
            'query_id': qid,
            'candidate_id': candidate_id,
            'evaluation': result,
            'score': score
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.config['output_path']), exist_ok=True)
        
        with open(self.config['output_path'], 'a', encoding='utf-8') as f:
            f.write(json.dumps(annotation_record, ensure_ascii=False) + '\n')

    def _print_final_statistics(self):
        print("\nAnnotation Statistics:")
        for qid, status in self.query_status.items():
            print(f"Query {qid}:")
            print(f"  Total annotations: {status['annotation_count']}")
            print(f"  Positive samples: {status['positive_samples']}")
            print(f"  Negative samples: {status['negative_samples']}")

    def generate_training_data(self):
        """Generate training data for reranker after annotation completion"""
        if not self.config['training_config']['generate_training_data']:
            print("Training data generation is disabled")
            return
            
        print("\nGenerating training data...")
        
        # Load all annotation results to analyze scores
        query_candidates = defaultdict(dict)
        
        try:
            with open(self.config['output_path'], 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    qid = record['query_id']
                    cid = record['candidate_id']
                    score = record.get('score')  # Use pre-calculated score if available
                    
                    if score is None:
                        # Calculate score if not available
                        score = self.calculate_score(record['evaluation']['dimension_scores'])
                    
                    query_candidates[qid][cid] = score
        except FileNotFoundError:
            print("No annotation results found")
            return
            
        # Generate training groups
        training_groups = []
        insufficient_samples = []
        
        output_path = self.config['training_config']['training_output_path']
        neg_per_group = self.config['training_config']['negative_samples_per_group']
        
        with open(output_path, 'w', encoding='utf-8') as fout:
            for qid, candidates in query_candidates.items():
                if qid not in self.query_record:
                    continue
                    
                # Separate positive and negative samples
                pos_samples = []  # score >= 2
                neg_samples = []  # score < 2 or None
                other_samples = []  # all samples for comparison
                
                for cid, score in candidates.items():
                    if score is None or score < 2:
                        neg_samples.append((cid, score if score is not None else -1))
                    else:
                        pos_samples.append((cid, score))
                        other_samples.append((cid, score))
                        
                if not pos_samples:
                    print(f"Query {qid} has no positive samples")
                    continue
                    
                # Sort by score (highest first)
                pos_samples.sort(key=lambda x: x[1], reverse=True)
                neg_samples.sort(key=lambda x: x[1], reverse=True)
                other_samples.sort(key=lambda x: x[1], reverse=True)
                
                # Create training groups for each positive sample
                for pos_cid, pos_score in pos_samples:
                    # Build negative sample pool
                    valid_negatives = []
                    
                    # First, add samples from other_samples with lower scores
                    valid_negatives.extend([(cid, score) for cid, score in other_samples 
                                          if score < pos_score and cid != pos_cid])
                    
                    # Then add negative samples
                    valid_negatives.extend(neg_samples)
                    
                    if len(valid_negatives) < neg_per_group:
                        insufficient_samples.append({
                            'query_id': qid,
                            'needed_negative': neg_per_group,
                            'available_negative': len(valid_negatives)
                        })
                        continue
                        
                    # Create training group in reranker format
                    training_group = {
                        'qry': {
                            'qid': str(qid),
                            'query': self.query_record[qid]
                        },
                        'pos': [{
                            'pid': pos_cid,
                            'passage': self.tuple_record[pos_cid]
                        }],
                        'neg': []
                    }
                    
                    # Add negative samples (with repetition if needed)
                    neg_idx = 0
                    for _ in range(neg_per_group):
                        neg_cid, _ = valid_negatives[neg_idx]
                        training_group['neg'].append({
                            'pid': neg_cid,
                            'passage': self.tuple_record[neg_cid]
                        })
                        neg_idx = (neg_idx + 1) % len(valid_negatives)
                    
                    print('--------------------------------')
                    print(f"Training group for query {qid}")
                    print(f"Positive: {[text['pid'] for text in training_group['pos']]}")
                    print(f"Negatives: {[text['pid'] for text in training_group['neg']]}")
                    
                    # Write training group to file
                    json.dump(training_group, fout, ensure_ascii=False)
                    fout.write('\n')
                    
                    training_groups.append(training_group)
        
        # Print training data statistics
        print(f"\nTraining Data Generation Complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total training groups generated: {len(training_groups)}")
        print(f"Queries processed: {len(set(g['qry']['qid'] for g in training_groups))}")
        
        if insufficient_samples:
            print(f"\nQueries with insufficient negative samples: {len(insufficient_samples)}")
            for item in insufficient_samples:
                print(f"  Query {item['query_id']}: needed {item['needed_negative']}, "
                      f"available {item['available_negative']}")
        
        return {
            'training_groups': training_groups,
            'insufficient_samples': insufficient_samples,
            'statistics': {
                'total_training_groups': len(training_groups),
                'queries_with_insufficient_negatives': len(insufficient_samples)
            }
        }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Construct training groups for reranking')
    
    # Data paths
    parser.add_argument('--collection', required=True, help='Path to collection.tsv file')
    parser.add_argument('--qrels', required=True, help='Path to qrels.tsv file')
    parser.add_argument('--folds', required=True, help='Path to folds.json file')
    parser.add_argument('--queries', required=True, help='Path to queries.tsv file')
    parser.add_argument('--retrieved_results', required=True, help='Path to retrieved results file')
    
    # OpenAI configuration
    parser.add_argument('--api_key', required=True, help='OpenAI API key')
    parser.add_argument('--api_base', required=True, help='OpenAI API base URL')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use')
    
    # Annotation configuration
    parser.add_argument('--max_annotations_per_query', type=int, default=30, 
                       help='Maximum annotations per query')
    parser.add_argument('--min_negative_samples', type=int, default=15,
                       help='Minimum negative samples required')
    parser.add_argument('--min_positive_samples', type=int, default=1,
                       help='Minimum positive samples required')
    
    # Training data generation configuration
    parser.add_argument('--generate_training_data', action='store_true',
                       help='Generate training data after annotation completion')
    parser.add_argument('--training_output_path', default=None,
                       help='Output path for training data (default: same dir as output_path)')
    parser.add_argument('--negative_samples_per_group', type=int, default=15,
                       help='Number of negative samples per training group')
    
    # Output
    parser.add_argument('--output_path', required=True, help='Output path for results')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    config = {
        'data_paths': {
            'collection': args.collection,
            'qrels': args.qrels,
            'folds': args.folds,
            'queries': args.queries,
            'retrieved_results': args.retrieved_results
        },
        'openai_config': {
            'api_key': args.api_key,
            'api_base': args.api_base,
            'model': args.model
        },
        'annotation_config': {
            'max_annotations_per_query': args.max_annotations_per_query,
            'min_negative_samples': args.min_negative_samples,
            'min_positive_samples': args.min_positive_samples
        },
        'training_config': {
            'generate_training_data': args.generate_training_data,
            'training_output_path': args.training_output_path or args.output_path.replace('.jsonl', '_training.jsonl'),
            'negative_samples_per_group': args.negative_samples_per_group
        },
        'output_path': args.output_path
    }

    annotator = DataAnnotator(config)
    annotator.annotate()

if __name__ == "__main__":
    main()