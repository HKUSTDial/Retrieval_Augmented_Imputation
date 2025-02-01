import argparse
import requests
import json
import jsonlines
import time
from collections import defaultdict
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

class DataImputer:
    def __init__(self, args):
        self.args = args
        self.template = """## Task
Assess whether missing values in the query table can be imputed using retrieved tables, ensuring the data is sufficient, relevant, and logically consistent:

1. **Examine the Input**: Review the query table with missing value and the provided "Retrieved Tables."
2. **Evaluate Retrieved Tables**: Use the following dimensions to assess the usefulness of each retrieved table for filling missing values in the query table:
* Existence: Does the tuple include at least one missing attribute? (Yes/No)
* Relevance: How related is the tuple to the query? (Highly Relevant/Somewhat Relevant/Not Relevant)
* Logical Consistency: Is the tuple logically aligned with the query (e.g., temporal consistency, dependencies)? (Fully Consistent/Partially Consistent/Not Consistent)

3. **Generate Response**: 
- If sufficient information exists: respond using JSON format: {answer_format} with confidence score （0-1）for each imputed attribute. The confidence level for each imputation is based on the reliability of the retrieved tuples and the rigor of the reasoning used to fill the missing values.
- If imputation is not feasible, return: "Sorry I can't provide the grounded imputed values since retrieved data do not contain useful information for imputation." 

Note: Only use information grounded in the retrieved tables. Do not rely on internal knowledge or unsupported reasoning.

## Input
Query: 
{question}

Retrieved Tables:
{retrieved_tables}
        """
        # 初始化API设置
        self.url = args.api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}"
        }
        
        # 加载数据
        self.load_data()
        
        self.lock = threading.Lock()  # 添加线程锁用于安全写入文件
        
    def load_data(self):
        """加载所需的所有数据文件"""
        # 加载检索结果
        self.topK_results = self.load_retrieval_results()
        # 加载数据集划分
        with open(self.args.folds_path, 'r') as f:
            self.test_qids = json.load(f)['test']

        print(f"test_qids: {len(self.test_qids)}")

        # 加载collection
        self.collection = self.load_collection()
        # 加载已处理的tuples
        self.processed_tuples = self.load_processed_tuples()
        
    def load_retrieval_results(self):
        """加载检索结果"""
        all_scores = defaultdict(dict)
        with open(self.args.retrieval_results_path, 'r') as f:
            for line in f:
                qid, docid, score = line.strip().split('\t')
                # qid, docid, rank, score = line.strip().split('\t')
                score = float(score)
                all_scores[qid][docid] = score
                
        # 获取top-k结果
        topK_results = {}
        for qid in all_scores:
            score_list = sorted(list(all_scores[qid].items()), 
                              key=lambda x: x[1], reverse=True)
            topK_results[qid] = [docid for docid, _ in score_list[:self.args.top_k]]
            
        return topK_results
    
    def load_collection(self):
        """加载collection数据"""
        collection = {}
        with open(self.args.collection_path, 'r') as f:
            for line in f:
                line = line.strip()
                qid, query = line[:line.find('\t')], line[line.find('\t')+1:]
                collection[qid] = query
        return collection
    
    def load_processed_tuples(self):
        """加载已处理的tuples"""
        processed_tuples = []
        if Path(self.args.output_path).exists():
            with jsonlines.open(self.args.output_path, 'r') as f:
                for line in f:
                    processed_tuples.append(int(line['tuple_id']))
        return processed_tuples

    def chat(self, input_data):
        """调用API进行对话"""
        data = {
            "model": self.args.model,
            "messages": [{"role": "user", "content": input_data}],
            "temperature": self.args.temperature,
            "logprobs": True
        }
        
        while True:
            try:
                response = requests.post(self.url, headers=self.headers, 
                                      data=json.dumps(data))
                response = response.json()
                break
            except:
                print("Error: requests.post")
                time.sleep(1)
                
        return (response['choices'][0]['message']['content'], 
                response['choices'][0]['logprobs'])
    
    def convert_to_table(self, tuple_id, serialized_tuple):
        """将序列化的tuple转换为表格格式"""
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
            
        table = f'caption: {title}\n|{" | ".join(headers)} |\n|{" | ".join(values)} |'
        return table
    
    def process_missing_tables(self):
        """使用多线程处理缺失值表格"""
        all_tasks = []
        with jsonlines.open(self.args.missing_tables_path, 'r') as f:
            for line in f:
                table_data = line
                tuple_ids = table_data['tuple_id']
                tableData = table_data['tableData']
                headers = table_data['processed_tableHeaders']
                caption = f"caption:{table_data['pgTitle']} | {table_data['sectionTitle']} | {table_data['tableCaption']}"
                
                for index, t_id in enumerate(tuple_ids):
                    # if (t_id in self.processed_tuples or 
                    #     t_id not in self.test_qids):
                    if t_id not in self.test_qids:
                        continue
                    # 将每个需要处理的tuple打包成任务
                    task = (headers, tableData[index], t_id, caption)
                    all_tasks.append(task)
        
        # 使用线程池处理任务
        with ThreadPoolExecutor(max_workers=self.args.num_threads) as executor:
            executor.map(self.process_single_tuple, all_tasks)
    
    def process_single_tuple(self, task):
        """处理单个tuple的线程函数"""
        headers, tuple_data, t_id, caption = task
        
        # 构建输入
        input_data = self.construct_input(headers, tuple_data, t_id, caption)
        # print(f"Processing tuple {t_id}")
        # print("---------------------------------------------------")
        # print(f"Input:\n{input_data}")
        
        # 获取输出
        output, logprobs = self.chat(input_data)
        # print(f"Output:\n{output}")

        # 使用线程锁安全地写入文件
        with self.lock:
            with jsonlines.open(self.args.output_path, 'a') as fout:
                fout.write({
                    'tuple_id': t_id,
                    'input': input_data,
                    'output': output,
                    'logprobs': logprobs
                })

    def construct_input(self, headers, tuple_data, t_id, caption):
        """构建模型输入"""
        # 构建question部分（包含caption和待填充的表格）
        question = caption + '\n'
        question += '|' + '|'.join(headers) + '|\n'
        
        # 构建answer_format和添加数据行
        answer_format = '{'
        row_data = []
        for j, value in enumerate(tuple_data):
            if value == 'N/A':
                answer_format += f'"{headers[j]}": {{"imputed_value": "", "confidence": ""}}, '
                row_data.append('[TO-FILL]')
            else:
                row_data.append(value)
                
        question += '|' + '|'.join(row_data) + '|\n'
        answer_format = answer_format[:-2] + '}'
        
        # 构建retrieved_tables部分
        retrieved_tables_text = ''
        retrieved_tables = self.topK_results[str(t_id)]
        for rank, docid in enumerate(retrieved_tables):
            retrieved_tables_text += f'Table {rank+1}: {self.convert_to_table(docid, self.collection[docid])}\n\n'
        
        # 使用format填充模板
        return self.template.format(
            answer_format=answer_format,
            question=question,
            retrieved_tables=retrieved_tables_text
        )

def main():
    parser = argparse.ArgumentParser()
    # API相关参数
    parser.add_argument('--api_url', type=str, default="https://vip.yi-zhan.top/v1/chat/completions")
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--model', type=str, default="gpt-4o-mini")
    parser.add_argument('--temperature', type=float, default=0.3)
    
    # 数据路径参数
    parser.add_argument('--missing_tables_path', type=str, required=True)
    parser.add_argument('--retrieval_results_path', type=str, required=True)
    parser.add_argument('--collection_path', type=str, required=True)
    parser.add_argument('--folds_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    # 其他参数
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_threads', type=int, default=4)
    
    args = parser.parse_args()
    
    imputer = DataImputer(args)
    imputer.process_missing_tables()

if __name__ == "__main__":
    main()
                