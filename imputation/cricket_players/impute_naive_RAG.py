import requests
import json
import jsonlines
import re
import time
import pandas as pd
import ast
from typing import Dict, Any, Optional
from collections import defaultdict
import os

import requests
import json
import jsonlines
import re
import time

url = "https://vip.yi-zhan.top/v1/chat/completions" 
openai_headers = { 
    "Content-Type": "application/json", 
    "Authorization": "Bearer sk-pP1nu21e5A9Ucg9bB82c0dC07d334b9aA7A0Bc26F3Eb2f72" 
} 


class CricketPlayersImputation:
    def __init__(self, 
                 model_name: str,
                 api_url: str,
                 api_key: str,
                 data_path: str = '/home/yangchenyu/Data_Imputation/data/cricket_players',
                 output_path: str = '/home/yangchenyu/Data_Imputation/imputation/results/cricket_players',
                 retrieval_file: str = None,
                 retriever: str = 'reranker',
                 top_k: int = 3):
        """
        初始化参数
        """
        self.model_name = model_name
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.data_path = data_path
        self.output_path = output_path
        self.retriever = retriever
        
        # 加载数据
        self.keep_columns = ['Name', 'Team', 'Type', 'ValueinCR', 'National Side', 
                           'Batting Style', 'Bowling', 'MatchPlayed', 'InningsBatted', 'RunsScored']
        self.df = pd.read_csv(f'{data_path}/cricket_ret_1.csv', usecols=self.keep_columns)
        
        with open(f'{data_path}/annotated_data/folds.json', 'r') as f:
            self.test_ids = json.load(f)['test']
            
        # 设置要补全的列
        self.missing_columns = ['National Side', 'Batting Style']
        
        # 设置检索文件路径
        self.top_k = top_k
        self.retrieval_file = retrieval_file or f'{self.data_path}/rerank_results/cricket_players_test.tsv'
        
        # 加载检索结果和集合数据
        self.topK_results = defaultdict(list)
        self.collection = {}
        self._load_retrieval_results()
        self._load_collection()
        
        # 更新模板
        self.template = '''Based on the retrieved tabular data and your own knowledge, what's the most likely value for the [TO-FILL] cell in the table below? Please respond using JSON: {answer_format}, the key is attribute name of each [TO-FILL], value is the predicted value for each [TO-FILL].\n'''
    

    def chat(self, input_data, model="gpt-4o-mini", temperate=0.8):
        
        data = { 
            "model": model, 
            "messages": [{"role": "user", "content": input_data}], 
            "temperature": temperate,
        } 
        # AttributeError: 'list' object has no attribute 'items'
        while(1):
            try:
                response = requests.post(url, headers=openai_headers, data=json.dumps(data)) 
                response = response.json()
                output_content = response['choices'][0]['message']['content']
                break
            except:
                print("Error: requests.post")
                time.sleep(1)


        return output_content


    def _load_retrieval_results(self):
        """加载检索结果，按分数排序后只保留top-k个结果"""
        temp_results = defaultdict(list)
        
        # 首先收集所有结果及其分数
        with open(self.retrieval_file, 'r') as f:
            for line in f:
                line = line.strip()
                # qid, docid, score = line.split('\t')
                qid, docid, rank, score = line.split('\t')
                qid = int(qid)
                docid = int(docid)
                score = float(score)
                temp_results[qid].append((docid, score))
        
        # 对每个查询的结果按分数排序并保留top-k
        for qid, results in temp_results.items():
            # 按分数降序排序
            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            # 只保留前k个文档id
            self.topK_results[qid] = [doc_id for doc_id, _ in sorted_results[:self.top_k]]

    
    def _load_collection(self):
        """加载集合数据"""
        with open(f'{self.data_path}/annotated_data/collection.tsv', 'r') as f:
            for line in f:
                line = line.strip()
                qid, query = line[:line.find('\t')], line[line.find('\t')+1:]
                self.collection[int(qid)] = query
    

    def convert_to_table(self, serialized_tuple):
        # 分割标题和数据
        caption_split = serialized_tuple.split(' attribute ')
        title = caption_split[0].split(']: ')[1].strip()

        # 提取属性和值
        attributes = caption_split[1:]

        headers = []
        values = []
        sign = 0
        

        for attribute in attributes:
            attribute_value_split = attribute.split(' value ')
            attribute_name = attribute_value_split[0].strip()
            value = attribute_value_split[1].split(' attribute ')[0].strip()  # 分割可能的下一个属性
            
            if sign == 1 and len(attribute_name) > 10:
                attribute_name = attribute_name[:10]
            headers.append(attribute_name)
            
            values.append(value)

        # 构建表格
        table = 'caption: ' + title + '\n|' + ' | '.join(headers) + ' |\n|' + ' | '.join(values) + ' |'
        return table


    def run_imputation(self, processed_tuples: list = None) -> float:
        """运行补全任务"""
        output_file = f'{self.output_path}/{self.model_name}_results_naive_RAG_{self.retriever}_top{self.top_k}.jsonl'
        print(output_file)
        # 读取已处理的数据
        processed_data = {}
        if os.path.exists(output_file):
            with jsonlines.open(output_file, 'r') as fin:
                for line in fin:
                    processed_data[line['tuple_id']] = line
            
        count, acc = 0, 0
        acc_by_column = defaultdict(int)  # 记录每个列的准确率
        
        for index, row in self.df.iterrows():
            if index not in self.test_ids:
                continue

            # 构建answer_format和添加数据行
            answer_format = '{'
            row_data = []
            ground_truths = {}
            
            for col in self.df.columns:
                if col in self.missing_columns:
                    cell_value = '[TO-FILL]'
                    answer_format += f'"{col}": "", '
                    if not pd.isnull(row[col]):
                        ground_truths[col] = row[col].lower().strip()
                else:
                    cell_value = row[col]
                row_data.append(str(cell_value))
            
            answer_format = answer_format[:-2] + '}'

            if index in processed_data:
                output = processed_data[index]['output']
                imputed_data = self.extract_imputed_value(output)
            else:

                input_data = self.template + '[caption]: cricket player\n'
                
                # 添加表头
                input_data += '|' + '|'.join(self.df.columns) + '|\n'
                
            
                input_data += '|' + '|'.join(row_data) + '|'
                input_data = input_data.format(answer_format=answer_format)

                # 添加检索到的表格
                input_data += '\nRetrieved Tables:\n'
                retrieved_tables = self.topK_results[index]
                for rank, docid in enumerate(retrieved_tables):
                    input_data += f'Table {rank+1}: {self.convert_to_table(self.collection[docid])}\n'

                print(f"\n处理第 {index} 条数据...")
                output = self.chat(input_data)
                imputed_data = self.extract_imputed_value(output)

                # 保存结果
                with jsonlines.open(output_file, 'a') as fout:
                    fout.write({
                        'tuple_id': index,
                        'input': input_data,
                        'output': output,
                        'ground_truth': ground_truths,
                        'prediction': imputed_data
                    })
                
            count += 1
            correct_predictions = 0

            if imputed_data:
                # 检查每个缺失列的预测
                for col in self.missing_columns:
                    if col in imputed_data and col in ground_truths:
                        imputed_value = imputed_data[col].lower().strip()
                        if imputed_value == ground_truths[col]:
                            acc_by_column[col] += 1
                            correct_predictions += 1
                        elif col == 'Batting Style':
                            if 'right' in imputed_value and 'right' in ground_truths[col]:
                                acc_by_column[col] += 1
                                correct_predictions += 1
                            elif 'left' in imputed_value and 'left' in ground_truths[col]:
                                acc_by_column[col] += 1
                                correct_predictions += 1
                

            acc += correct_predictions / len(self.missing_columns)  # 计算平均准确率

        # 计算总体结果
        accuracy = acc / count if count > 0 else 0
        print(f"\n总体准确率: {accuracy:.4f} ({acc}/{count})")
        
            
        return accuracy
    

    def extract_imputed_value(self, output: str) -> Optional[Dict[str, Any]]:
        """从输出中提取预测值"""
        try:
            # 匹配```json和```之间的内容
            json_pattern = r'```json\s*({[^`]*})\s*```'
            match = re.search(json_pattern, output, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                return ast.literal_eval(json_str)
            else:
                return None
        except:
            return None


    def extract_results_from_file(self, result_file: str) -> pd.DataFrame:
        """从结果文件中提取预测值并生成DataFrame"""
        results = []
        
        with jsonlines.open(result_file, 'r') as fin:
            for line in fin:
                tuple_id = line['tuple_id']
                prediction = line.get('prediction', {})
                ground_truth = line.get('ground_truth', {})
                
                result = {'tuple_id': tuple_id}
                
                # 添加预测值
                for col in self.missing_columns:
                    result[f'{col}_pred'] = prediction.get(col, '')
                    result[f'{col}_true'] = ground_truth.get(col, '')
                
                results.append(result)
        
        return pd.DataFrame(results)


imputer = CricketPlayersImputation(
    model_name="gpt-4o",
    api_url="https://vip.yi-zhan.top/v1/chat/completions",
    api_key="sk-pP1nu21e5A9Ucg9bB82c0dC07d334b9aA7A0Bc26F3Eb2f72",
    retrieval_file="/home/yangchenyu/Data_Imputation/retrieval_results/first_stage/BM25_top100_res_with_score_cricket_players.tsv",
    top_k=5,
    retriever='BM25'
)

# 运行补全任务
accuracy = imputer.run_imputation()


# # 提取结果到DataFrame进行进一步分析
# results_df = imputer.extract_results_from_file(result_file)
# print(results_df.head())

