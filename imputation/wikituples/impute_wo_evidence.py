import argparse
import requests
import json
import jsonlines
import re
import time
import numpy as np
import random
from typing import Dict, List, Any

class TableImputer:
    def __init__(self, args):
        self.api_url = args.api_url
        self.api_key = args.api_key
        self.model = args.model
        
        self.entity_vocab = None
        self.entityid_to_text = {}
        self.all_entity_set = set()

        self.temperature = args.temperature
        self.output_path = args.output_path
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
    def chat(self, input_data: str) -> str:
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": input_data}],
            "temperature": self.temperature
        }
        
        while True:
            try:
                response = requests.post(self.api_url, headers=self.headers, data=json.dumps(data))
                response = response.json()
                break
            except:
                print("Error: API request failed, retrying...")
                time.sleep(1)
        
        return response['choices'][0]['message']['content']

    def _load_queries(self, query_file: str) -> Dict[str, str]:
        queries = {}
        with open(query_file, 'r') as f:
            for line in f:
                line = line.strip()
                qid, query = line[:line.find('\t')], line[line.find('\t')+1:]
                queries[qid] = query
        print(f"Loaded {len(queries)} queries")
        return queries

    def _load_entity_vocab(self, vocab_file: str, min_ent_count: int = 2, ignore_bad_title: bool = True) -> Dict:
        entity_vocab = {}
        bad_title = 0
        few_entity = 0
        
        with open(vocab_file, 'r', encoding="utf-8") as f:
            for line in f:
                _, entity_id, entity_title, mid, count = line.strip().split('\t')
                if ignore_bad_title and entity_title == '':
                    bad_title += 1
                elif int(count) < min_ent_count:
                    few_entity += 1
                else:
                    entity_vocab[len(entity_vocab)] = {
                        'wiki_id': int(entity_id),
                        'wiki_title': entity_title,
                        'count': count
                    }
        
        print(f'Total number of entity: {len(entity_vocab)}')
        print(f'Remove because of empty title: {bad_title}')
        print(f'Remove because count<{min_ent_count}: {few_entity}')
        
        # 构建实体ID到文本的映射
        self.all_entity_set = set([item['wiki_id'] for _,item in entity_vocab.items()])
        for _,item in entity_vocab.items():
            self.entityid_to_text[item['wiki_id']] = [item['wiki_title']]
            
        return entity_vocab
    
    def _load_processed_tuples(self) -> List[int]:
        processed_tuples = []
        try:
            with open(self.output_path, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    tuple_id = int(line['tuple_id'])
                    processed_tuples.append(tuple_id)
        except FileNotFoundError:
            print(f"No existing processed tuples file found at {self.output_path}")
        print(f"Found {len(processed_tuples)} processed tuples")
        return processed_tuples
    
    def _prepare_input_data(self, template: str, caption: str, table: Dict, index: int) -> str:
        headers = table['processed_tableHeaders']
        tuple_d = table['tableData'][index]
        tuple_ids = table['tuple_id']
        ground_truth = table['ground_truth']
        t_id = tuple_ids[index]
        
        input_data = template + caption + '\n'
        
        # Add headers
        for header in headers:
            input_data += '|' + header
        input_data += '|\n'
        
        # Add example tuple if available
        try:
            new_tuple_ids = [tt for tt in tuple_ids if tt != t_id]
            selected_tuple = random.choice(new_tuple_ids)
            example_tuple = ground_truth[tuple_ids.index(selected_tuple)]
            
            for value in example_tuple:
                if isinstance(value, list):
                    input_data += '|' + value[-1]
                else:
                    input_data += '|' + value
            input_data += '|\n'
        except:
            print("No other examples available")
        # Prepare current tuple and answer format
        answer_format = '{'
        for j, value in enumerate(tuple_d):
            if value == 'N/A':
                answer_format += f'"{headers[j]}": "", '
                input_data += '|[TO-FILL]'
            else:
                input_data += '|' + value
        
        input_data += '|\n'
        answer_format = answer_format[:-2] + '}'
        
        return input_data.format(answer_format=answer_format)

    def load_data(self, data_dir: str):
        """加载所有必要的数据文件"""
        print("Loading data...")
        # 加载查询数据
        self.queries = self._load_queries(f"{data_dir}/final_data/queries.tsv")

        with open(f"{data_dir}/final_data/folds.json", 'r') as f:
            self.folds = json.load(f)
        self.queries = [self.queries[str(test_id)] for test_id in self.folds['test']]
        # 加载实体词汇表
        self.entity_vocab = self._load_entity_vocab(f"{data_dir}/entity_vocab.txt", 
                                                  min_ent_count=2, 
                                                  ignore_bad_title=True)
        
        # 加载训练表格数据，更新实体文本映射
        self._load_train_tables(f"{data_dir}/train_tables.jsonl")
        
        # 加载已处理的元组
        self.processed_tuples = self._load_processed_tuples()
        
    def _load_train_tables(self, train_file: str):
        """加载训练表格并更新实体文本映射"""
        print(f"Loading training tables from {train_file}")
        with jsonlines.open(train_file, 'r') as f:
            for table in f:
                rows = table.get("tableData", {})
                for row in rows:
                    for cell in row:
                        surface_links = cell.get('surfaceLinks', [])
                        if surface_links and surface_links[0]['target']['id'] in self.all_entity_set:
                            entity_id = surface_links[0]['target']['id']
                            self.entityid_to_text[entity_id].append(cell['text'])
        
        print(f"Updated entity text mappings for {len(self.entityid_to_text)} entities")

    def process_tables(self, input_file: str):
        template = '''What's the most likely value for the [TO-FILL] cell in the table below? Please respond using JSON: {answer_format}, the key is attribute name of each [TO-FILL], value is the predicted value for each [TO-FILL].\n'''
        
        with jsonlines.open(input_file, 'r') as f:
            for table in f:
                self._process_single_table(table, template)
                
    def _process_single_table(self, table: Dict, template: str):
        table_id = table['_id']
        caption = f"caption:{table['pgTitle']} | {table['sectionTitle']} | {table['tableCaption']}"
        
        for index, t_id in enumerate(table['tuple_id']):
            if t_id in self.processed_tuples:
                continue
                
            print(f"\nProcessing tuple {t_id} from table {table_id}")
            input_data = self._prepare_input_data(template, caption, table, index)
            
            print("---------------------------------------------------")
            print(f"Input:\n{input_data}")
            
            output = self.chat(input_data)
            print(f"Output:\n{output}")
            
            self._save_result(t_id, input_data, output)
            
    def _save_result(self, tuple_id: int, input_data: str, output: str):
        with jsonlines.open(self.output_path, 'a') as fout:
            fout.write({
                'tuple_id': tuple_id,
                'input': input_data,
                'output': output
            })

def main():
    parser = argparse.ArgumentParser(description='Table Imputation without Evidence')
    parser.add_argument('--api_url', type=str, required=True, help='API endpoint URL')
    parser.add_argument('--api_key', type=str, required=True, help='API key')
    parser.add_argument('--model', type=str, default='gpt-4', help='Model to use')
    parser.add_argument('--temperature', type=float, default=0.3, help='Temperature for generation')
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory path')
    parser.add_argument('--input_file', type=str, required=True, help='Input tables file path')
    parser.add_argument('--output_path', type=str, required=True, help='Output file path')
    
    args = parser.parse_args()
    
    imputer = TableImputer(args)
    imputer.load_data(args.data_dir)
    imputer.process_tables(args.input_file)

if __name__ == "__main__":
    main()



