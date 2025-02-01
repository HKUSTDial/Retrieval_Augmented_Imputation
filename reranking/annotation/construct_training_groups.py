import json
import re
from collections import defaultdict
import openai

# 保留原有的OpenAI设置
openai.api_key = "sk-xxxx"
openai.api_base = "xxxxxx"

gpt_model = "gpt-4o-mini"

def generate_response(messages):
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
        temperature=0.1,
        max_tokens=500
    )
    return response.choices[0].message.content

# 新的prompt模板
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
    

    def extract_responses(text):
        """从GPT响应中提取各个维度的评分"""
        responses = {
            'existence': None,
            'relevance': None,
            'consistency': None
        }
        
        # 使用正则表达式提取[Response]标签中的内容
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
        
        response = generate_response(messages)
        dimension_scores = extract_responses(response)
        
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
                    
                    # 更新query状态
                    self.query_status[qid]['annotation_count'] += 1
                    self.query_status[qid]['processed_candidates'].add(candidate_id)  # 记录已处理的candidate
                    
                    # Calculate score and determine if it's positive or negative sample
                    score = self.calculate_score(record['evaluation']['dimension_scores'])
                    if score is not None and score >= 2:
                        self.query_status[qid]['positive_samples'] += 1
                    else:
                        self.query_status[qid]['negative_samples'] += 1
        except FileNotFoundError:
            print("未找到已有的评估结果文件,将创建新文件")

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
        for qid in self.train_qids:
            if self.should_stop_query(qid):
                print(f"Query {qid} completed processing, skipping")
                continue

            print(f'Processing query {qid}: {self.query_record[qid]}')
            
            for candidate_id in self.retrieved_results[qid]:
                if self.should_stop_query(qid):
                    break
                    
                if candidate_id in self.query_status[qid]['processed_candidates']:
                    continue
                
                self._process_candidate(qid, candidate_id)
                
        self._print_final_statistics()

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

    def _print_final_statistics(self):
        print("\nAnnotation Statistics:")
        for qid, status in self.query_status.items():
            print(f"Query {qid}:")
            print(f"  Total annotations: {status['annotation_count']}")
            print(f"  Positive samples: {status['positive_samples']}")
            print(f"  Negative samples: {status['negative_samples']}")

# Example usage:
if __name__ == "__main__":
    config = {
        'data_paths': {
            'collection': '/path/to/collection.tsv',
            'qrels': '/path/to/qrels.tsv',
            'folds': '/path/to/folds.json',
            'queries': '/path/to/queries.tsv',
            'retrieved_results': '/path/to/retrieved_results.tsv'
        },
        'openai_config': {
            'api_key': 'sk-xxxx',
            'api_base': 'xxxxxx',
            'model': 'gpt-4o-mini'
        },
        'annotation_config': {
            'max_annotations_per_query': 30,
            'min_negative_samples': 15,
            'min_positive_samples': 1
        },
        'output_path': './show_movie/dimension_evaluation_results.jsonl'
    }

    annotator = DataAnnotator(config)
    annotator.annotate()