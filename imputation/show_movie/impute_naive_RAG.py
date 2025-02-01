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

def chat(input_data, model="gpt-4o-mini", temperate=0.8):
    
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

class ShowMovieImputation:
    def __init__(self, 
                 model_name: str,
                 api_url: str,
                 api_key: str,
                 data_path: str = '/home/yangchenyu/Data_Imputation/data/show_movie',
                 output_path: str = '/home/yangchenyu/Data_Imputation/imputation/results/show_movie',
                 retrieval_file: str = None,
                 top_k: int = 3):
        """
        初始化参数
        model_name: 使用的模型名称
        api_url: API接口地址
        api_key: API密钥
        data_path: 数据集路径
        output_path: 输出结果路径
        retrieval_file: 检索结果文件路径
        top_k: 限制使用的检索结果数量
        """
        self.model_name = model_name
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.data_path = data_path
        self.output_path = output_path
        
        # 加载数据
        self.df = pd.read_csv(f'{data_path}/tv_ret_1.csv')
        with open(f'{data_path}/annotated_data/folds.json', 'r') as f:
            self.test_ids = json.load(f)['test']
            
        # 示例数据
        self.example = ['Avatar: The Last Airbender', '2005', 'TV-Y7', '9.2', '1']
        
        # 设置检索文件路径
        self.top_k = top_k
        self.retrieval_file = retrieval_file or f'{self.data_path}/rerank_results/show_movie_test.tsv'
        
        # 加载检索结果和集合数据
        self.topK_results = defaultdict(list)
        self.collection = {}
        self._load_retrieval_results()
        self._load_collection()
        
        # 更新模板
        self.template = '''Based on the retrieved tabular data, what's the most likely value for the [TO-FILL] cell in the table below? Please respond using JSON: {answer_format}, the key is attribute name of each [TO-FILL], value is the predicted value for each [TO-FILL].\n'''

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

    def convert_to_table(self, serialized_tuple: str) -> str:
        """将序列化的元组转换为表格格式"""
        caption_split = serialized_tuple.split(' attribute ')
        title = caption_split[0].split(']: ')[1].strip()
        
        attributes = caption_split[1:]
        headers = []
        values = []
        
        for attribute in attributes:
            attribute_value_split = attribute.split(' value ')
            attribute_name = attribute_value_split[0].strip()
            value = attribute_value_split[1].split(' attribute ')[0].strip()
            headers.append(attribute_name)
            values.append(value)
            
        return f'caption: {title}\n|{" | ".join(headers)} |\n|{" | ".join(values)} |'

    def chat(self, input_data: str, temperature: float = 0.3) -> str:
        """调用API获取响应"""
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": input_data}],
            "temperature": temperature,
        }
        
        while True:
            try:
                response = requests.post(self.api_url, headers=self.headers, data=json.dumps(data))
                response = response.json()
                return response['choices'][0]['message']['content']
            except:
                print("Error: API请求失败，1秒后重试")
                time.sleep(1)

    def extract_imputed_value(self, output: str) -> Optional[Dict[str, Any]]:
        """从输出中提取预测值"""
        try:
            output = output.replace('```json', '').replace('```', '')
            return ast.literal_eval(output)
        except:
            return None

    def run_imputation(self, processed_tuples: list = None) -> float:
        """运行补全任务
        processed_tuples: 已处理过的tuple_ids列表，可选
        返回准确率
        """
        # 获取已处理的tuple_ids
        output_file = f'{self.output_path}/{self.model_name}_results_naive_RAG_BM25_top{self.top_k}.jsonl'
        processed_tuples = set()
        
        # 读取已有的结果文件，获取已处理的tuple_ids
        if os.path.exists(output_file):
            with jsonlines.open(output_file, 'r') as fin:
                for line in fin:
                    processed_tuples.add(line['tuple_id'])
            print(f"已找到{len(processed_tuples)}条处理过的记录")
            
        count, acc = 0, 0
        
        # 处理未完成的数据
        for index, row in self.df.iterrows():
            if index not in self.test_ids or index in processed_tuples:
                continue

            # 构建输入
            input_data = self.template + '[caption]: tv shows-movies\n'
            
            # 添加表头
            input_data += '|' + '|'.join(self.df.columns) + '|\n'
            
            # 添加示例
            input_data += '|' + '|'.join(str(x) for x in self.example) + '|\n'
            
            # 添加待补全数据
            row_data = []
            ground_truth = None
            for col in self.df.columns:
                if col == 'Rating':
                    cell_value = '[TO-FILL]'
                    ground_truth = row[col].lower().strip()
                else:
                    cell_value = row[col]
                row_data.append(str(cell_value))
            
            input_data += '|' + '|'.join(row_data) + '|'
            input_data = input_data.format(answer_format="{Rating: ''}")

            # 在构建输入时添加检索到的表格
            input_data += '\nRetrieved Tables:\n'
            retrieved_tables = self.topK_results[index]
            for rank, docid in enumerate(retrieved_tables):
                input_data += f'Table {rank+1}: {self.convert_to_table(self.collection[docid])}\n'

            print(f"\n处理第 {index} 条数据...")
            output = self.chat(input_data)
            imputed_data = self.extract_imputed_value(output)
            
            count += 1

            if imputed_data and 'Rating' in imputed_data:
                imputed_value = imputed_data['Rating'].lower().strip()
                if imputed_value in ground_truth:
                    acc += 1

                # 保存结果
                with jsonlines.open(output_file, 'a') as fout:
                    fout.write({
                        'tuple_id': index,
                        'input': input_data,
                        'output': output,
                        'ground_truth': ground_truth,
                        'prediction': imputed_value
                    })

        # 计算总体结果
        total_count, total_acc = 0, 0
        with jsonlines.open(output_file, 'r') as fin:
            for line in fin:
                total_count += 1
                if line['prediction'] in line['ground_truth']:
                    total_acc += 1

        accuracy = total_acc/total_count if total_count > 0 else 0
        print(f"\n当前批次准确率: {acc/count if count > 0 else 0:.4f} ({acc}/{count})")
        print(f"总体准确率: {accuracy:.4f} ({total_acc}/{total_count})")
        return accuracy



imputer = ShowMovieImputation(
    model_name="gpt-4o-mini",
    api_url="https://vip.yi-zhan.top/v1/chat/completions",
    api_key="OPENAI_API_KEY",
    retrieval_file="/home/yangchenyu/Data_Imputation/retrieval_results/first_stage/BM25_top100_res_with_score_show_movie.tsv",  # 指定检索文件路径
    top_k=5  # 指定top_k参数
)

# 运行补全任务
accuracy = imputer.run_imputation()
