import argparse
import json
import jsonlines
import time
import numpy as np
from collections import defaultdict
from pathlib import Path
import openai
import math
import threading
from concurrent.futures import ThreadPoolExecutor
import re
import ast






class Imputer:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.results = []
        
        # Store specified missing columns if provided
        self.specified_missing_columns = self.args.missing_columns
        
        # Verification result cache
        self.verification_cache_file = Path(str(Path(self.args.output_path).parent) + "/verification_cache.json")
        self.verification_cache = self.load_verification_cache()
        
        # Define prompt templates for both modes
        self.strict_template = """## Task
Assess whether missing values in the query table can be imputed using retrieved tables, ensuring the data is sufficient, relevant, and logically consistent:

1. **Examine the Input**: Review the query table with missing value and the provided "Retrieved Tables."
2. **Evaluate Retrieved Tables**: Use the following dimensions to assess the usefulness of each retrieved table for filling missing values in the query table:
* Existence: Does the tuple include at least one missing attribute? (Yes/No)
* Relevance: How related is the tuple to the query? (Highly Relevant/Somewhat Relevant/Not Relevant)
* Logical Consistency: Is the tuple logically aligned with the query (e.g., temporal consistency, dependencies)? (Fully Consistent/Partially Consistent/Not Consistent)

3. **Generate Response**: 
- If sufficient information exists, respond using JSON format: {answer_format}
- For each imputed value, provide a confidence score (0-1) based on:
  * 1.0: Only when there is direct, explicit evidence in the retrieved tables with perfect match
  * 0.8-0.9: Strong evidence with minor variations or multiple consistent sources
  * 0.5-0.7: Moderate evidence with some uncertainty
  * <0.5: Weak or indirect evidence with significant uncertainty
- If imputation is not feasible, return: "Sorry I can't provide the grounded imputed values since retrieved data do not contain useful information for imputation."

{address_instructions}

Note: Only use information grounded in the retrieved tables. Do not rely on internal knowledge or unsupported reasoning.

## Input
Query: 
{question}

Retrieved Tables:
{retrieved_tables}
        """

        self.relaxed_template = """Based on the retrieved tabular data and your own knowledge, what's the most likely value for the missing cells in the table below? Please respond using JSON: {answer_format}. For each missing attribute (key in the JSON), provide both the `imputed_value` and a `confidence` score between 0 and 1 reflecting your certainty in the imputed value.

{address_instructions}

Query Table:
{question}

Retrieved Tables:
{retrieved_tables}"""

        # Setup OpenAI and load data
        self.setup_openai()
        self.load_data()

    def setup_openai(self):
        """Setup OpenAI API configuration"""
        openai.api_key = self.args.api_key
        openai.api_base = self.args.api_url
        self.model_name = self.args.model
    

    def convert_to_table(self, serialized_tuple):
        """Convert serialized tuple to table format"""
        try:
            caption_split = serialized_tuple.split(' attribute ')
            if len(caption_split) < 2:
                print(f"Warning: Unable to parse tuple: {serialized_tuple[:100]}...")
                return f"Error parsing: {serialized_tuple[:50]}..."
                
            # Extract title
            title_parts = caption_split[0].split(']: ')
            if len(title_parts) < 2:
                title = caption_split[0].strip()
            else:
                title = title_parts[1].strip()
            
            attributes = caption_split[1:]
            
            headers = []
            values = []
            
            for attribute in attributes:
                try:
                    attribute_value_split = attribute.split(' value ')
                    if len(attribute_value_split) < 2:
                        # Handle improperly formatted attributes
                        print(f"Warning: Incorrect attribute format: {attribute}")
                        headers.append(attribute.strip())
                        values.append("N/A")
                        continue
                        
                    attribute_name = attribute_value_split[0].strip()
                    
                    # Safely extract attribute values
                    remaining_parts = attribute_value_split[1].split(' attribute ')
                    value = remaining_parts[0].strip()
                    
                    headers.append(attribute_name)
                    values.append(value)
                except Exception as e:
                    print(f"Error parsing attribute: {str(e)}, skipping: {attribute[:50]}...")
                    continue
                
            if not headers or not values:
                return f"Empty table from: {serialized_tuple[:50]}..."
                
            table = f'caption: {title}\n|{" | ".join(headers)} |\n|{" | ".join(values)} |'
            return table
            
        except Exception as e:
            print(f"Error converting to table: {str(e)}")
            return f"Error converting: {serialized_tuple[:50]}..."
    

    def save_verification_cache(self):
        """Save verification result cache"""
        try:
            with open(self.verification_cache_file, 'w') as f:
                json.dump(self.verification_cache, f)
            print(f"Verification cache saved with {len(self.verification_cache)} records")
        except Exception as e:
            print(f"Error saving verification cache: {e}")


    def load_data(self):
        """Load required data files"""
        # Load fold data to get train tuple IDs
        with open(f'{self.args.data_path}/folds.json', 'r') as f:
            self.test_qids = json.load(f)['test']
        print(f"Test set size: {len(self.test_qids)}")

        # Load queries
        self.queries = {}
        with open(f'{self.args.data_path}/queries.tsv', 'r') as f:
            for line in f:
                line = line.strip()
                qid, query = int(line[:line.index('\t')]), line[line.index('\t')+1:]
                self.queries[qid] = self.convert_to_table(query)
                
        # Load ground truth answers
        self.ground_truth = {}
        with jsonlines.open(f'{self.args.data_path}/answers.jsonl', 'r') as f:
            for line in f:
                self.ground_truth[line['query_id']] = line['answers']
                
        # Load collection
        self.collection = {}
        with open(f'{self.args.data_path}/collection.tsv', 'r') as f:
            for line in f:
                line = line.strip()
                qid, query = line[:line.find('\t')], line[line.find('\t')+1:]
                self.collection[int(qid)] = query
                
        # Load retrieval results
        self.topK_results = self.load_retrieval_results()

    def load_retrieval_results(self):
        """Load retrieval results"""
        all_scores = defaultdict(dict)
        with open(self.args.retrieval_results_path, 'r') as f:
            for line in f:
                qid, docid, score = line.strip().split('\t')
                all_scores[int(qid)][int(docid)] = float(score)
        
        topK_results = defaultdict(list)
        for qid in all_scores:
            sorted_results = sorted(all_scores[qid].items(), 
                                key=lambda x: x[1], 
                                reverse=True)
            for docid, _ in sorted_results[:self.args.top_k]:
                topK_results[int(qid)].append(int(docid))
        
        return topK_results

    def get_processed_queries(self):
        """Get processed query IDs"""
        processed_qids = set()
        if Path(self.args.output_path).exists():
            with jsonlines.open(self.args.output_path, 'r') as f:
                for line in f:
                    if 'query_id' in line:
                        processed_qids.add(line['query_id'])
        return processed_qids
    
    
    def extract_attributes(self, serialized_tuple):
        """Extract attributes and values from table-format tuple"""
        attributes = {}
        
        try:
            # Split header and data rows
            lines = serialized_tuple.strip().split('\n')
            if len(lines) < 3:
                print(f"Warning: Insufficient table rows: {serialized_tuple}")
                return {}
            
            # Extract headers and values
            try:
                headers = [h.strip() for h in lines[1].strip('|').split('|')]
                values = [v.strip() for v in lines[2].strip('|').split('|')]
                
                # Ensure headers and values have consistent length
                if len(headers) != len(values):
                    print(f"Warning: Mismatch in number of headers and values: {len(headers)} vs {len(values)}")
                    # Adjust length to match
                    min_len = min(len(headers), len(values))
                    headers = headers[:min_len]
                    values = values[:min_len]
                
                # Combine into dictionary
                for header, value in zip(headers, values):
                    attributes[header.strip()] = value.strip()
                
                return attributes
            except Exception as e:
                print(f"Error extracting attributes: {str(e)}")
                return {}
                
        except Exception as e:
            print(f"Error parsing table: {str(e)}")
            return {}

    def get_missing_columns(self, tuple_data):
        """Identify columns that need to be filled"""
        # If missing columns are specified through args, use those
        if self.specified_missing_columns is not None:
            # Return only the columns that exist in the tuple_data
            return [col for col in self.specified_missing_columns if col in tuple_data]
            
        # Otherwise, identify columns with N/A values
        missing_cols = []
        for header, value in tuple_data.items():
            # Check if it's N/A
            if value.lower() in ['n/a']:
                missing_cols.append(header)
        return missing_cols


    def impute_value(self, qid, mode='strict'):
        """Perform missing value imputation using specified mode"""
        tuple_data = self.queries[qid]
        attributes = self.extract_attributes(tuple_data)
        missing_columns = self.get_missing_columns(attributes)
       
        if not missing_columns:
            return None, None, None, None, None
            
        # Construct input
        input_data = self.construct_input(qid, missing_columns, mode)
        
        # Get output
        output, logprobs = self.chat(input_data)
        
        # Extract predictions and confidence scores
        predictions, confidences = self.extract_prediction(output, logprobs, missing_columns, mode)
        
        return predictions, confidences, input_data, output, logprobs
    
    def process_query(self, qid):

        """Process single query, including both strict and relaxed modes"""
        # First use strict mode
        strict_predictions, strict_confidences, strict_input, strict_output, strict_logprobs = self.impute_value(qid, 'strict')
        
        print("="*50)
        print( strict_predictions, strict_confidences)
        print("="*30)
        print(strict_input, strict_output )

        if not strict_predictions:
            return
            
        # Initialize final results
        final_predictions = {}
        final_confidences = {}
        modes_used = {}
        
        # Initialize relaxed mode related variables
        relaxed_predictions = None
        relaxed_confidences = None
        relaxed_input = None
        relaxed_output = None
        relaxed_logprobs = None
        
        # Process each missing column
        for col in strict_predictions:
            if col not in strict_confidences['llm']:
                continue
                
            confidence = strict_confidences['llm'][col]
            
            # If strict mode confidence is above threshold, use strict results
            if confidence >= self.args.threshold:
                final_predictions[col] = strict_predictions[col]
                final_confidences[col] = confidence
                modes_used[col] = 'strict'
            else:
                # If relaxed mode hasn't been run yet, run it now
                if relaxed_predictions is None:
                    relaxed_predictions, relaxed_confidences, relaxed_input, relaxed_output, relaxed_logprobs = self.impute_value(qid, 'relaxed')
                
                if relaxed_predictions and col in relaxed_predictions:
                    final_predictions[col] = relaxed_predictions[col]
                    final_confidences[col] = relaxed_confidences['llm'][col]
                    modes_used[col] = 'relaxed'
                else:
                    # If relaxed mode also fails, use strict mode results
                    final_predictions[col] = strict_predictions[col]
                    final_confidences[col] = confidence
                    modes_used[col] = 'strict_fallback'
        
        
        # Record results
        result = {
            'query_id': qid,
            'strict_mode': {
                'input': strict_input,
                'output': strict_output,
                'predictions': strict_predictions,
                'confidences': strict_confidences,
                'logprobs': strict_logprobs,
                'correctness': self.verify_predictions(qid, strict_predictions)
            }, 
            'relaxed_mode': {
                'input': relaxed_input,
                'output': relaxed_output,
                'predictions': relaxed_predictions,
                'confidences': relaxed_confidences,
                'logprobs': relaxed_logprobs,
                'correctness': self.verify_predictions(qid, relaxed_predictions) if relaxed_predictions else None
            },
            'final_predictions': final_predictions,
            'final_confidences': final_confidences,
            'ground_truth': self.ground_truth.get(qid, {}),
            'modes_used': modes_used,
            'final_correctness': self.verify_predictions(qid, final_predictions)
        }
        
        # Safely write to file
        with self.lock:
            with jsonlines.open(self.args.output_path, 'a') as fout:
                fout.write(result)
    
    
    
    def chat(self, input_data: str, temperature: float = 0.3, model_name: str = None) -> tuple:
        """Get response and logprobs using OpenAI API"""

        messages = [{"role": "user", "content": input_data}]
        
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name if model_name is None else model_name,
                    messages=messages,
                    temperature=temperature,
                    logprobs=True,
                    top_logprobs=5
                )
               
                return (response['choices'][0]['message']['content'], response['choices'][0]['logprobs'])
            
            except Exception as e:
                print(f"Error: API request failed: {str(e)}, retrying in 1 second")
                time.sleep(1)


    def extract_prediction(self, output, logprobs, missing_columns, filling_mode):
        """Extract predictions and multiple confidence scores from model output"""

        json_pattern = r'```json\s*({[^`]*})\s*```'
        match = re.search(json_pattern, output, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    result = ast.literal_eval(json_str)
                except:
                    return None, None
        else:
            return None, None
        
        predictions = {}
        confidences = {
            'llm': {},  # LLM generated confidence
            'nln': {},  # Negative length-normalized score
            'max_entropy': {}  # Maximum entropy based
        }
        
        # Extract predictions and LLM-generated confidence scores
        for col in missing_columns:
            if col in result:
                if isinstance(result[col], dict):
                    # Get imputed_value, ensure string type handling
                    imputed_value = result[col].get('imputed_value', '')
                    if isinstance(imputed_value, (int, float)):
                        predictions[col] = str(imputed_value).strip()
                    else:
                        predictions[col] = imputed_value.lower().strip()
                    
                    # Handle empty confidence values or invalid confidence values
                    confidence_value = result[col].get('confidence', 0)
                    if confidence_value == "" or confidence_value is None:
                        confidences['llm'][col] = 0.0  # Set default confidence to 0
                    else:
                        try:
                            confidences['llm'][col] = float(confidence_value)
                        except (ValueError, TypeError):
                            print(f"Warning: Column '{col}' confidence value '{confidence_value}' cannot be converted to float, using default value 0.0")
                            confidences['llm'][col] = 0.0
                else:
                    # Handle non-dictionary direct values
                    if isinstance(result[col], (int, float)):
                        predictions[col] = str(result[col]).strip()
                    else:
                        predictions[col] = str(result[col]).lower().strip()
                    confidences['llm'][col] = 1.0
        
        # Extract required information from logprobs
        if not logprobs or 'content' not in logprobs or not logprobs['content']:
                print("Warning: Logprobs are missing or empty.")
                # Assign default low confidence if logprobs are unavailable
                for col in missing_columns:
                    if col in predictions:
                        confidences['nln'][col] = 0.0
                        confidences['max_entropy'][col] = 0.0
                return predictions, confidences # Return early if no logprobs

        token_logprobs = [item['logprob'] for item in logprobs['content']]
        tokens = [item['token'] for item in logprobs['content']]
        top_logprobs_list = []
        for item in logprobs['content']:
            # Extract logprob values from each token's top_logprobs
            token_top_logprobs = {}
            for top_item in item['top_logprobs']:
                token_top_logprobs[top_item['token']] = top_item['logprob']
            top_logprobs_list.append(token_top_logprobs)
        
        # Calculate other confidence scores based on filling_mode
        if filling_mode == 'strict':
            # Strict mode: use overall statistical features, all missing values share the same confidence
            overall_nln = self.length_normalized_confidence(token_logprobs)
            overall_entropy = self.calculate_entropy(top_logprobs_list)
            
            for col in missing_columns:
                if col in predictions:
                    confidences['nln'][col] = overall_nln
                    confidences['max_entropy'][col] = overall_entropy
                
        else:  # relaxed mode
            # Relaxed mode: locate token positions corresponding to each column's JSON entry
            processed_cols = set()  # Track processed columns
            
            for col in missing_columns:
                if col not in predictions:
                    continue

                col_logprobs = []
                col_top_logprobs = []
                
                # --- Improved column value range search logic ---
                target_key_pattern = f'"{col}"'

                # 1. First try to exactly match the entire column name (including quotes)
                key_indices = []
                for i, token in enumerate(tokens):
                    if target_key_pattern in token:
                        key_indices.append(i)
                    # Check if multiple tokens need to be combined
                    elif i < len(tokens) - 10:  # Ensure enough tokens to combine
                        combined = ''.join(tokens[i:i+10])  # Try combining up to 4 tokens
                        if target_key_pattern in combined:
                            key_indices.append(i)
                
                if not key_indices:
                    print(f"Column '{col}' key name not found, will use overall logprobs")
                    confidences['nln'][col] = self.length_normalized_confidence(token_logprobs)
                    confidences['max_entropy'][col] = self.calculate_entropy(top_logprobs_list)
                    continue
                
                # Start search from the last occurrence of key_index, which is usually the actual column name position in JSON
                found_range = False
                key_idx = key_indices[-1]  # Use the last found key index
                
                # 2. First find the colon ":"
                colon_idx = -1
                for i in range(key_idx + 1, min(key_idx + 10, len(tokens))):
                    if ':' in tokens[i]:
                        colon_idx = i
                        break
                
                if colon_idx == -1:
                    print(f"Column '{col}' no colon found after key name, will use overall logprobs")
                    confidences['nln'][col] = self.length_normalized_confidence(token_logprobs)
                    confidences['max_entropy'][col] = self.calculate_entropy(top_logprobs_list)
                    continue
                
                # 3. Find the first left brace "{" after the colon
                start_value_idx = -1
                for i in range(colon_idx + 1, min(colon_idx + 15, len(tokens))):
                    if '{' in tokens[i]:
                        start_value_idx = i
                        break
                
                if start_value_idx == -1:
                    print(f"Column '{col}' no left brace found after colon, will use overall logprobs")
                    confidences['nln'][col] = self.length_normalized_confidence(token_logprobs)
                    confidences['max_entropy'][col] = self.calculate_entropy(top_logprobs_list)
                    continue
                
                # 4. Track nesting from left brace until matching right brace is found
                brace_level = 1  # Already found one left brace
                end_value_idx = -1
                
                for i in range(start_value_idx + 1, len(tokens)):
                    for char in tokens[i]:
                        if char == '{':
                            brace_level += 1
                        elif char == '}':
                            brace_level -= 1
                            # Found matching right brace
                            if brace_level == 0:
                                end_value_idx = i
                                break
                    
                    if brace_level == 0:
                        break
                
                if end_value_idx == -1:
                    print(f"Column '{col}' could not find matching right brace, will use overall logprobs")
                    confidences['nln'][col] = self.length_normalized_confidence(token_logprobs)
                    confidences['max_entropy'][col] = self.calculate_entropy(top_logprobs_list)
                    continue
                
                # 5. Verify if the found range contains expected content
                range_content = ' '.join(tokens[key_idx:end_value_idx+1])

                # Check if range content contains keywords
                if 'confidence' in range_content and 'im puted _value' in range_content:
                    found_range = True
                    
                    # Find actual start position of column name (may be before key_idx)
                    actual_start_key_idx = key_idx
                    for i in range(key_idx, max(0, key_idx-5), -1):
                        if '"' in tokens[i]:
                            actual_start_key_idx = i
                            break
                    
                    # Extract logprobs within the range
                    col_indices = list(range(actual_start_key_idx, end_value_idx + 1))
                    col_logprobs = [token_logprobs[i] for i in col_indices]
                    col_top_logprobs = [top_logprobs_list[i] for i in col_indices]
                    
                    print(f"Column '{col}' range: from {actual_start_key_idx} to {end_value_idx}")
                    print(f"Range contains tokens: {' '.join(tokens[actual_start_key_idx:end_value_idx+1])}")
            
                # Calculate confidence for this column
                if found_range and col_logprobs:
                    confidences['nln'][col] = self.length_normalized_confidence(col_logprobs)
                    confidences['max_entropy'][col] = self.calculate_entropy(col_top_logprobs)
                else:
                    # All methods failed, use overall logprobs
                    print(f"Warning: Unable to determine specific token range for column '{col}'. Using overall logprobs.")
                    confidences['nln'][col] = self.length_normalized_confidence(token_logprobs)
                    confidences['max_entropy'][col] = self.calculate_entropy(top_logprobs_list)
        
        return predictions, confidences
    

    def length_normalized_confidence(self, log_probs):
        """
        Calculate length-normalized confidence score ∈ [0,1] based on token log probabilities
        Parameters:
            log_probs: List[float] — log P (log probability) for each token

        Returns:
            confidence: float — Length-normalized confidence score, higher values indicate more reliability
        """
        L = len(log_probs)
        confidence = np.exp(sum(log_probs) / L)  # This is equivalent to \tilde{P}(s|x) in the formula
        return confidence

    def calculate_entropy(self, top_logprobs_list):
        """
        Calculate average entropy based on token top-K log probabilities and map to confidence score
        Args:
            top_logprobs_list: List[Dict[str, float]] top-K log probs for each position
        Returns:
            confidence: float ∈ [0,1]
        """
        entropies = []

        for logprobs in top_logprobs_list:
            if not logprobs:
                continue

            # Convert to probabilities
            probs = np.array([math.exp(lp) for lp in logprobs.values()])
            probs /= probs.sum()  # Ensure normalization to distribution

            # Calculate entropy H(p) = -sum(p log p)
            entropy = -np.sum(probs * np.log(probs + 1e-10))  # Prevent log(0)
            entropies.append(entropy)

        if not entropies:
            return 0.0  # If no valid data, return lowest confidence

        avg_entropy = np.mean(entropies)

        # Map to confidence [0,1], lower entropy → higher confidence
        confidence = np.exp(-avg_entropy)

        return confidence

            
    def construct_input(self, qid, missing_columns, filling_mode):
        """Construct model input based on filling mode"""
        # Add address-specific instructions for restaurants dataset
        address_instructions = ""
        if self.args.dataset == 'restaurants':
            address_instructions = """\nFor address fields, provide complete and specific address including building number, building/complex name, street name, area/locality, city, etc. (e.g., "18, B K Complex, Opposite To BDA Complex, 21st Main Road, 2nd Stage, Banashankari, Bangalore").\n"""
        
        if filling_mode == 'relaxed':
            return self.construct_naive_input(qid, missing_columns, address_instructions)
        else:
            # Original detailed analysis mode
            question = self.queries[qid]
            
            # Build answer_format
            answer_format = '{'
            for col in missing_columns:
                answer_format += f'"{col}": {{"imputed_value": "", "confidence": ""}}, '
            answer_format = answer_format[:-2] + '}'
            
            # Get retrieval results
            retrieved_tables_text = ''
            if qid in self.topK_results:
                for rank, docid in enumerate(self.topK_results[qid]):
                    retrieved_tables_text += f'Table {rank+1}: {self.convert_to_table(self.collection[docid])}\n\n'

            return self.strict_template.format(
                answer_format=answer_format,
                question=question,
                retrieved_tables=retrieved_tables_text,
                address_instructions=address_instructions
            )

    
    def construct_naive_input(self, qid, missing_columns, address_instructions=""):
        """Construct model input for naive filling mode"""
        question = self.queries[qid]
        
        # Build answer format
        answer_format = '{'
        for col in missing_columns:
            answer_format += f'"{col}": {{"imputed_value": "", "confidence": ""}}, '
        answer_format = answer_format[:-2] + '}'
        
        # Get retrieved tables
        retrieved_tables_text = ''
        if qid in self.topK_results:
            for rank, docid in enumerate(self.topK_results[qid][:5]):
                retrieved_tables_text += f'Table {rank+1}: {self.convert_to_table(self.collection[docid])}\n\n'
        
        return self.relaxed_template.format(
            answer_format=answer_format,
            question=question,
            retrieved_tables=retrieved_tables_text,
            address_instructions=address_instructions
        )

    def verify_predictions(self, qid, predictions):
        """Verify correctness of prediction results"""
        correctness = {}
        ground_truth = self.ground_truth.get(qid, {})
        
        for col, prediction in predictions.items():
            if col not in ground_truth:
                continue

            if self.args.dataset == 'wikituples' or self.args.dataset == 'restaurants' or (self.args.dataset == 'education' and col == 'Street Address'):
                is_correct = self.verify_entity_equivalence(prediction, ground_truth[col], qid, col)
            elif self.args.dataset == 'cricket_players' and col == 'Batting Style':
                is_correct = 0
                if 'right' in prediction.lower() and 'right' in ground_truth[col][0].lower():
                    is_correct = 1
                if 'left' in prediction.lower() and 'left' in ground_truth[col][0].lower():
                    is_correct = 1
            else:
                if isinstance(ground_truth[col], str):
                    is_correct = 1 if prediction.lower().strip() in ground_truth[col].lower().strip() else 0
                elif isinstance(ground_truth[col], list):
                    is_correct = 1 if any(prediction.lower().strip() in gold.lower().strip() for gold in ground_truth[col]) else 0
            
                
            correctness[col] = is_correct
            
        return correctness


    def verify_entity_equivalence(self, prediction, answer_set, t_id, hint):
        """Use LLM to determine if prediction and any item in the answer set refer to the same entity"""
        
        # Create cache key
        cache_key = f"{prediction}_|_|_{str(answer_set)}"
        
        # Check cache
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key]
        
        # If not in cache, use LLM verification
        # Prepare the prompt for LLM
        prompt = f"""Strictly determine if the prediction and any item in the answer set refer to EXACTLY the same entity/concept. They should describe the identical entity with only minor textual variations (e.g., formatting, capitalization, spacing).
    
For addresses, the prediction must match the EXACT address in the answer set - matching only city/state/province is insufficient. The addresses must refer to the same specific location.

Examples of equivalent pairs:
- "New York City" = "NYC"
- "United States of America" = "USA"
- "William Shakespeare" = "Shakespeare, William"

Examples of non-equivalent pairs:
- "New York City" ≠ "New York State"
- "Apple Inc." ≠ "Apple Computer"
- "John F. Kennedy" ≠ "John Kennedy Jr."

Prediction: {prediction}
Answer Set: {answer_set}

Answer only 'Yes' or 'No'."""


        response, _ = self.chat(prompt, model_name='gpt-4o-mini')
        result = 1 if 'yes' in response.lower() else 0
        
        # Save to cache
        self.verification_cache[cache_key] = result
        
        # Save cache every 10 verifications
        if len(self.verification_cache) % 10 == 0:
            self.save_verification_cache()
            
        return result


    def load_verification_cache(self):
        """Load verification result cache"""
        if self.verification_cache_file.exists():
            try:
                with open(self.verification_cache_file, 'r') as f:
                    cache = json.load(f)
                print(f"Loaded verification cache with {len(cache)} records")
                return cache
            except Exception as e:
                print(f"Error loading verification cache: {e}")
                return {}
        return {}

    def run(self):
        """Run imputation program"""
        # Get processed queries
        processed_qids = self.get_processed_queries()
        print(f"Number of processed queries: {len(processed_qids)}")
        
        # Get remaining queries to process
        remaining_qids = [qid for qid in self.test_qids if qid not in processed_qids]
        print(f"Number of remaining queries to process: {len(remaining_qids)}")
        
        for qid in remaining_qids:
            self.process_query(qid)
            
        # Save verification cache
        self.save_verification_cache()

        analysis_results = self.analyze_results()
        
        return analysis_results

    def analyze_results(self):
        """Analyze various metrics of imputation results"""
        print("\nStarting analysis of imputation results...")
        
        # Initialize counters
        total_predictions = 0
        correct_predictions = 0
        strict_predictions = 0
        strict_correct = 0
        total_missing_values = 0
        
        
        # Read all results
        with jsonlines.open(self.args.output_path, 'r') as f:
            for line in f:
                # Get strict mode results
                strict_mode = line['strict_mode']
                strict_preds = strict_mode['predictions']
                strict_confs = strict_mode['confidences']
                strict_correctness = strict_mode['correctness']
                
                # Get final results
                final_preds = line['final_predictions']
                # If final_correctness doesn't exist, recalculate
                if 'final_correctness' not in line:
                    final_correctness = self.verify_predictions(line['query_id'], final_preds)
                else:
                    final_correctness = line['final_correctness']
                    
                modes_used = line.get('modes_used', {})
                ground_truth = line['ground_truth']
                
                # Count total missing values
                for col in ground_truth:
                    if col in final_preds:  # Only count values that were attempted to be filled
                        total_missing_values += 1
                        
                        # Count overall accuracy
                        if col in final_correctness:
                            total_predictions += 1
                            correct_predictions += final_correctness[col]
                        
                        # Count strict mode accuracy
                        if modes_used.get(col) == 'strict':
                            strict_predictions += 1
                            if col in strict_correctness:
                                strict_correct += strict_correctness[col]
        
        # Calculate metrics
        overall_precision = correct_predictions / total_predictions if total_predictions > 0 else 0
        strict_precision = strict_correct / strict_predictions if strict_predictions > 0 else 0
        strict_ratio = strict_predictions / total_missing_values if total_missing_values > 0 else 0
        
        print(f"\n=== Imputation Results Analysis ===")
        print(f"Threshold: {self.args.threshold}")
        print(f"Overall Accuracy: {overall_precision:.4f} ({correct_predictions}/{total_predictions})")
        print(f"Strict Mode Accuracy: {strict_precision:.4f} ({strict_correct}/{strict_predictions})")
        print(f"Strict Mode Usage Ratio: {strict_ratio:.4f} ({strict_predictions}/{total_missing_values})")
        
        # Save analysis results
        output_dir = str(Path(self.args.output_path).parent)
        analysis_file = f'{output_dir}/analysis_threshold_{self.args.threshold}.json'
        
        with open(analysis_file, 'w') as f:
            json.dump({
                'threshold': self.args.threshold,
                'overall_precision': overall_precision,
                'strict_precision': strict_precision,
                'strict_ratio': strict_ratio,
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'strict_predictions': strict_predictions,
                'strict_correct': strict_correct,
                'total_missing_values': total_missing_values
            }, f, indent=2)
        
        print(f"\nAnalysis results saved to: {analysis_file}")
        
        return {
            'overall_precision': overall_precision,
            'strict_precision': strict_precision,
            'strict_ratio': strict_ratio
        }

    def evaluate_only(self):
        """Evaluate existing results without processing new data."""
        # Get processed queries
        processed_qids = self.get_processed_queries()
        print(f"Number of processed queries: {len(processed_qids)}")

        # Save verification cache
        self.save_verification_cache()

        analysis_results = self.analyze_results()
        
        return analysis_results
    

def main():
    parser = argparse.ArgumentParser()
    # API related parameters
    parser.add_argument('--api_url', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--model', type=str, default="gpt-4")
    
    # Data path parameters
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--retrieval_results_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    # Other parameters
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.8,
                      help='Confidence threshold for using strict mode results')
    parser.add_argument('--num_threads', type=int, default=4)
    # Add new argument for specifying missing columns
    parser.add_argument('--missing_columns', type=str, nargs='+', default=None,
                      help='Specify columns to impute. If not provided, will detect N/A values')
    
    # Add an argument for evaluation only mode
    parser.add_argument('--evaluation_only', action='store_true',
                        help='Run in evaluation only mode without processing new data')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Run the imputer
    imputer = Imputer(args)
    
    if args.evaluation_only:
        imputer.evaluate_only()
    else:
        imputer.run()

if __name__ == "__main__":
    main()                  