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

url = "https://vip.yi-zhan.top/v1/chat/completions" 
openai_headers = { 
    "Content-Type": "application/json", 
    "Authorization": "Bearer OPENAI_API_KEY" 
} 


class EducationImputation:
    def __init__(self, 
                 model_name: str,
                 api_url: str,
                 api_key: str,
                 data_path: str = '/home/yangchenyu/Data_Imputation/data/education',
                 output_path: str = '/home/yangchenyu/Data_Imputation/imputation/results/education',
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
        with open(f'{data_path}/annotated_data/queries.tsv', 'r') as f:
            self.queries = {}
            for line in f:
                line = line.strip()
                qid, query = int(line[:line.index('\t')]), line[line.index('\t')+1:]
                self.queries[qid] = self.convert_to_table(query)
        
        with open(f'{data_path}/annotated_data/folds.json', 'r') as f:
            self.test_ids = json.load(f)['test']
            
        # 加载正确答案
        self.ground_truth = {}
        with jsonlines.open(f'{data_path}/annotated_data/answer.jsonl', 'r') as f:
            for line in f:
                self.ground_truth[line['query_id']] = line['answers']
            
        # 设置要补全的列
        self.missing_columns = ['Street Address', 'ZIP Code', 'Phone Number']
        
        
        # 设置检索文件路径
        self.top_k = top_k
        self.retrieval_file = retrieval_file or f'{self.data_path}/rerank_results/education_test.tsv'
        
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

        for attribute in attributes:
            attribute_value_split = attribute.split(' value ')
            attribute_name = attribute_value_split[0].strip()
            value = attribute_value_split[1].split(' attribute ')[0].strip()  # 分割可能的下一个属性
            headers.append(attribute_name)
            values.append(value)

        # 构建表格
        table = 'caption: ' + title + '\n|' + ' | '.join(headers) + ' |\n|' + ' | '.join(values) + ' |'
        return table.replace('N/A', '[TO-FILL]')



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
        
        for qid, query in self.queries.items():
            if qid not in self.test_ids:
                continue

            # 构建answer_format
            answer_format = '{'
            for col in self.missing_columns:
                answer_format += f'"{col}": "", '
            answer_format = answer_format[:-2] + '}'

            if qid in processed_data:
                output = processed_data[qid]['output']
                imputed_data = self.extract_imputed_value(output)
            else:

                input_data = self.template + query + '\n'
                input_data = input_data.format(answer_format=answer_format)

                # 添加检索到的表格
                input_data += '\nRetrieved Tables:\n'
                retrieved_tables = self.topK_results[qid]
                for rank, docid in enumerate(retrieved_tables):
                    input_data += f'Table {rank+1}: {self.convert_to_table(self.collection[docid])}\n'

                print(f"\n处理第 {qid} 条数据...")
                # print(input_data)
                output = self.chat(input_data)
                # print(output)
                imputed_data = self.extract_imputed_value(output)

                # 保存结果
                with jsonlines.open(output_file, 'a') as fout:
                    fout.write({
                        'tuple_id': qid,
                        'input': input_data,
                        'output': output,
                        'prediction': imputed_data
                    })
                
            count += len(self.missing_columns)
            
            if imputed_data:
                # 获取正确答案
                correct_values = {}
                for key, value in self.ground_truth[qid].items():
                    correct_values[key] = []
                    for vv in value:
                        correct_values[key].append(vv.lower().replace('(','').replace(')',''))

                # 检查每个缺失列的预测
                for col in self.missing_columns:
                    if col in imputed_data:
                        if imputed_data[col].lower().replace('(','').replace(')','') in correct_values[col]:
                            acc_by_column[col] += 1
                            acc += 1
                        else:
                            print(f"预测错误: {col} = {imputed_data[col]}")
                            print(f"正确答案: {correct_values[col]}")
                            print(f"填充: {imputed_data[col]}")

        # 计算总体结果
        accuracy = acc / count if count > 0 else 0
        print(f"\n总体准确率: {accuracy:.4f} ({acc}/{count})")
        
        # 打印每个列的准确率
        for col in self.missing_columns:
            col_accuracy = acc_by_column[col] / len(self.test_ids) if len(self.test_ids) > 0 else 0
            print(f"{col} 准确率: {col_accuracy:.4f} ({acc_by_column[col]}/{len(self.test_ids)})")
            
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


imputer = EducationImputation(
    model_name="gpt-4o-mini",
    api_url="https://vip.yi-zhan.top/v1/chat/completions",
    api_key="sk-pP1nu21e5A9Ucg9bB82c0dC07d334b9aA7A0Bc26F3Eb2f72",
    retrieval_file="/home/yangchenyu/Data_Imputation/retrieval_results/first_stage/BM25_top100_res_with_score_education.tsv",
    top_k=5,
    retriever='BM25'
)

# 运行补全任务
accuracy = imputer.run_imputation()

