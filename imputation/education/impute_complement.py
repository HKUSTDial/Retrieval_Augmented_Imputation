import argparse
import requests
import json
import jsonlines
import time
import pandas as pd
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading
import re
import ast
import sys
from typing import Optional, Dict, Any
import os

class EducationImputation:
    def __init__(self, 
                 model_name: str,
                 api_url: str,
                 api_key: str,
                 data_path: str = '/home/yangchenyu/Data_Imputation/data/education',
                 output_path: str = '/home/yangchenyu/Data_Imputation/imputation/results/education',
                 retrieval_file: str = None,
                 retriever: str = 'BM25',
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
        
        # 设置检索文件路径
        self.top_k = top_k
        self.retrieval_file = retrieval_file or f'{self.data_path}/rerank_results/education_test.tsv'
        
        # 加载检索结果和集合数据
        self.topK_results = defaultdict(list)
        self.collection = {}


        # 加载数据
        with open(f'{data_path}/annotated_data/queries.tsv', 'r') as f:
            self.queries = {}
            for line in f:
                line = line.strip()
                qid, query = int(line[:line.index('\t')]), line[line.index('\t')+1:]
                self.queries[qid] = self.convert_to_table(query)
        
        # 设置要补全的列
        self.missing_columns = ['Street Address', 'ZIP Code', 'Phone Number']

        self._load_retrieval_results()
        self._load_collection()
        self.load_processed_tuples()
        
        # 更新模板
        self.template = '''Based on the retrieved tabular data and your own knowledge, what's the most likely value for the [TO-FILL] cell in the table below? Please respond using JSON: {answer_format}, the key is attribute name of each [TO-FILL], value is the predicted value for each [TO-FILL].\n'''
    
    def _load_retrieval_results(self):
        """加载检索结果，按分数排序后只保留top-k个结果"""
        temp_results = defaultdict(list)
        
        # 首先收集所有结果及其分数
        with open(self.retrieval_file, 'r') as f:
            for line in f:
                line = line.strip()
                qid, docid, score = line.split('\t')
                # qid, docid, rank, score = line.split('\t')
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
    
    def load_processed_tuples(self, processed_tuples: list = None) -> float:
        """运行补全任务
        processed_tuples: 已处理过的tuple_ids列表，可选
        返回准确率
        """
        # 获取已处理的tuple_ids
        output_file = f'{self.output_path}/{self.model_name}_results_naive_RAG_{self.retriever}_top{self.top_k}.jsonl'
        self.results = {}
        
        # 读取已有的结果文件，获取已处理的tuple_ids
        if os.path.exists(output_file):
            with jsonlines.open(output_file, 'r') as fin:
                for line in fin:
                    imputed_data = self.extract_imputed_value(line['output'])
                    self.results[line['tuple_id']] = imputed_data
            print(f"已找到{len(self.results)}条处理过的记录")
    
    def run_imputation(self, tuple_id):
        """运行单个补全任务并保存结果
        
        Args:
            tuple_id: 需要补全的数据ID
        Returns:
            dict: 预测结果
        """
        # 检查是否已经有结果
        if tuple_id in self.results:
            return self.results[tuple_id]
            
        # 构建输入
        query = self.queries[tuple_id]
        
        # 构建answer_format
        answer_format = '{'
        for col in self.missing_columns:
            answer_format += f'"{col}": "", '
        answer_format = answer_format[:-2] + '}'
        
        # 获取检索结果
        retrieved_tables_text = ''
        if tuple_id in self.topK_results:
            for rank, docid in enumerate(self.topK_results[tuple_id]):
                retrieved_tables_text += f'Table {rank+1}: {self.convert_to_table(self.collection[docid])}\n\n'
        
        # 构建完整输入
        input_data = self.template + query + '\n'
        input_data = input_data.format(answer_format=answer_format)

        # 添加检索到的表格
        input_data += '\nRetrieved Tables:\n' + retrieved_tables_text

        # 获取模型输出
        output = self.chat(input_data)
        # 提取预测结果
        prediction = self.extract_imputed_value(output)
        # 保存结果
        result = {
            'tuple_id': tuple_id,
            'input': input_data,
            'output': output,
            'prediction': prediction
        }
        
        # 更新内存中的结果
        self.results[tuple_id] = prediction
        
        # 保存到文件
        result_file = f'{self.output_path}/{self.model_name}_results_naive_RAG_{self.retriever}_top{self.top_k}.jsonl'
        with jsonlines.open(result_file, 'a') as fout:
            fout.write(result)
            
        return prediction

class EducationImputer:
    def __init__(self, args):
        self.args = args
        self.template = """## Task
Assess whether missing values in the query table can be imputed using retrieved tables, ensuring the data is sufficient, relevant, and logically consistent:

1. **Examine the Input**: Review the query table with missing value and the provided "Retrieved Tables."
2. **Evaluate Retrieved Tables**: Use the following dimensions to assess the usefulness of each retrieved table for filling missing values in the query table:
* Existence: Does the tuple include at least one attribute to be filled? (Yes/No)
* Relevance: How related is the tuple to the query? (Highly Relevant/Somewhat Relevant/Not Relevant)
* Logical Consistency: Is the tuple logically aligned with the query? (Fully Consistent/Partially Consistent/Not Consistent)

3. **Generate Response**: 
- Based on the above analysis, you MUST attempt to fill the missing values
- Respond in JSON format: {answer_format}
- For each imputed value, provide:
  * The imputed value
  * A confidence score (0-1) based on the quality and reliability of the source information
- Only if NO relevant information exists in ANY of the retrieved tables, then respond: "Sorry I can't provide the grounded imputed values since retrieved data do not contain useful information for imputation."

## Input
Query: 
{question}

Retrieved Tables:
{retrieved_tables}
        """
        
        self.url = args.api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}"
        }
        self.missing_columns = ['Street Address', 'ZIP Code', 'Phone Number']
        self.results = []
        self.lock = threading.Lock()
        
        # 加载数据
        self.load_data()
        
        # 初始化naive RAG实例
        self.naive_rag = EducationImputation(
            model_name=args.model,
            api_url=args.api_url,
            api_key=args.api_key,
            retriever=args.retriever,
            retrieval_file=args.retrieval_results_path,
            top_k=args.top_k
        )
        
        # 加载已有的预测结果
        self.existing_predictions = self.load_existing_predictions()
    
    def load_data(self):
        """加载所需的所有数据文件"""
        # 加载检索结果
        self.topK_results = self.load_retrieval_results()
        
        # 加载数据集划分
        with open(self.args.folds_path, 'r') as f:
            self.test_qids = json.load(f)['test']
        print(f"测试集大小: {len(self.test_qids)}")

        # 加载collection
        self.collection = self.load_collection()
        
        # 加载queries
        self.queries = {}
        with open(f'{self.args.data_path}/annotated_data/queries.tsv', 'r') as f:
            for line in f:
                line = line.strip()
                qid, query = int(line[:line.index('\t')]), line[line.index('\t')+1:]
                self.queries[qid] = self.convert_to_table(query)
        
        # 加载ground truth
        self.ground_truth = {}
        with jsonlines.open(f'{self.args.data_path}/annotated_data/answer.jsonl', 'r') as f:
            for line in f:
                self.ground_truth[line['query_id']] = line['answers']

    def load_retrieval_results(self):
        """加载检索结果"""
        all_scores = defaultdict(dict)
        
        # 首先加载所有检索结果和分数
        with open(self.args.retrieval_results_path, 'r') as f:
            for line in f:
                qid, docid, score = line.strip().split('\t')
                # qid, docid, rank, score = line.strip().split('\t')
                score = float(score)
                all_scores[qid][docid] = score
        
        # 对每个查询按分数排序并获取top-k结果
        topK_results = defaultdict(list)
        for qid in all_scores:
            # 按分数降序排序
            sorted_results = sorted(all_scores[qid].items(), 
                                key=lambda x: x[1], 
                                reverse=True)
            # 取top-k
            for docid, _ in sorted_results[:self.args.top_k]:
                topK_results[int(qid)].append(int(docid))
        
        print(f"已加载检索结果，共 {len(topK_results)} 个查询")
        return topK_results
        
    def load_collection(self):
        """加载collection数据"""
        collection = {}
        with open(self.args.collection_path, 'r') as f:
            for line in f:
                line = line.strip()
                qid, query = line[:line.find('\t')], line[line.find('\t')+1:]
                collection[int(qid)] = query
        return collection
    
    def load_processed_tuples(self):
        """加载已处理的结果和tuple IDs"""
        processed_tuples = []
        
        # 如果输出文件存在，加载已有的结果
        if Path(self.args.output_path).exists():
            print(f"发现已有的预测结果文件: {self.args.output_path}")
            try:
                loaded_count = 0
                with jsonlines.open(self.args.output_path, 'r') as f:
                    for line in f:
                        processed_tuples.append(int(line['tuple_id']))
                        self.results.append(line)
                        loaded_count += 1
                
                print(f"成功加载 {loaded_count} 条预测结果")
                
                # 如果指定了立即评估标志，进行评估
                if hasattr(self.args, 'evaluate_only') and self.args.evaluate_only:
                    if hasattr(self.args, 'thresholds') and self.args.thresholds:
                        self.evaluate_multiple_thresholds(self.args.thresholds)
                    else:
                        self.evaluate()
                    # 评估完成后退出程序
                    sys.exit(0)
                    
            except Exception as e:
                print(f"加载已有结果时出错: {str(e)}")
                print("将重新开始处理...")
                self.results = []  # 清空结果
                processed_tuples = []
                
        return processed_tuples

    def load_existing_predictions(self):
        """加载已有的预测结果"""
        predictions = {}
        if Path(self.args.output_path).exists():
            print(f"加载已有预测结果: {self.args.output_path}")
            with jsonlines.open(self.args.output_path, 'r') as f:
                for line in f:
                    predictions[line['tuple_id']] = {'ground_truth': line['ground_truth'], 'prediction': line['prediction'], 'confidence': line['confidence']}
            print(f"成功加载 {len(predictions)} 条预测结果")
        return predictions

    def chat(self, input_data):
        """调用API进行对话"""
        data = {
            "model": self.args.model,
            "messages": [{"role": "user", "content": input_data}],
            "temperature": self.args.temperature,
        }
        
        while True:
            try:
                response = requests.post(self.url, headers=self.headers, 
                                      data=json.dumps(data))
                response = response.json()
                return response['choices'][0]['message']['content']
            except:
                print("Error: API请求失败，1秒后重试")
                time.sleep(1)
    
    def convert_to_table(self, serialized_tuple):
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
    
    
    def extract_prediction(self, output):
        """从模型输出中提取预测值和置信度"""
        try:
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
                json_start = output.find('{')
                json_end = output.rfind('}')
                if json_start != -1 and json_end != -1:
                    json_str = output[json_start:json_end + 1]
                    try:
                        return self.extract_prediction(f"```json{json_str}```")
                    except:
                        return None, None
                return None, None
            
            # 提取预测值和置信度
            predictions = {}
            confidences = {}
            
            for col in self.missing_columns:
                if col in result:
                    if isinstance(result[col], dict):
                        predictions[col] = result[col].get('imputed_value', '').lower().strip()
                        confidences[col] = float(result[col].get('confidence', 0))
                    else:
                        predictions[col] = str(result[col]).lower().strip()
                        confidences[col] = 1.0
                        
            return predictions, confidences
            
        except Exception as e:
            return None, None

    def construct_input(self, qid):
        """构建模型输入"""
        question = self.queries[qid]
        
        # 构建answer_format
        answer_format = '{'
        for col in self.missing_columns:
            answer_format += f'"{col}": {{"imputed_value": "", "confidence": ""}}, '
        answer_format = answer_format[:-2] + '}'
        
        retrieved_tables_text = ''
        if qid in self.topK_results:
            for rank, docid in enumerate(self.topK_results[qid]):
                retrieved_tables_text += f'Table {rank+1}: {self.convert_to_table(self.collection[docid])}\n\n'
        
        return self.template.format(
            answer_format=answer_format,
            question=question,
            retrieved_tables=retrieved_tables_text
        )

    def evaluate_with_naive_rag_backup(self, threshold=None,):
        """使用naive RAG作为后备的评估方法"""
        if not self.existing_predictions:
            print("没有找到已有的预测结果！")
            return
        
        print(f"开始评估 {len(self.existing_predictions)} 条预测结果...")
        
        
        # 初始化总体指标
        total_samples = 0
        total_correct = 0
        filled_count = 0
        
        for tuple_id, result in self.existing_predictions.items():
            # print(result)
            original_predictions = result['prediction']
            confidences = result['confidence']
            ground_truths = result['ground_truth']
            
            final_predictions = {}
            used_naive_rag = {}
            is_correct = {}
            
            for attr in self.missing_columns:
                if attr in ground_truths:
                    total_samples += 1

                    try:
                        orig_pred = original_predictions.get(attr)
                        confidence = confidences.get(attr)
                    except:
                        orig_pred = None
                        confidence = None
                    
                    # 判断是否需要使用naive RAG
                    need_backup = (threshold is not None and confidence is not None and confidence < threshold) or orig_pred is None

                    filled_count += 1
                    
                    final_pred = None
                    if need_backup:
                        # 获取或运行naive RAG的预测
                        if tuple_id not in self.naive_rag.results:
                            naive_prediction = self.naive_rag.run_imputation(tuple_id)
                        else:
                            naive_prediction = self.naive_rag.results[tuple_id]
                        
                        
                        
                        if naive_prediction and attr in naive_prediction:
                            prediction = naive_prediction[attr].lower().strip()
                            if prediction in [ans.lower().strip() for ans in ground_truths[attr]]:
                                total_correct += 1
                        else:
                            final_pred = None
                            is_correct[attr] = False
                            
                        used_naive_rag[attr] = True
                            
                    else:
                        final_pred = orig_pred
                        # print('************************')
                        # print(final_pred, ground_truths[attr])

                        if final_pred in [ans.lower().strip() for ans in ground_truths[attr]]:
                            total_correct += 1
                        else:
                            final_pred = None
                            is_correct[attr] = False
                        
                                
                        used_naive_rag[attr] = False
                        
                    final_predictions[attr] = final_pred
            
        # 计算总体准确率
        overall_accuracy = total_correct / total_samples
        
        # 打印总体评估结果
        print(f"\n总体评估结果 (threshold={threshold}):")
        print(f"总样本数: {total_samples}")
        print(f"总填充数: {filled_count}")
        print(f"总正确预测数: {total_correct}")
        print(f"总体准确率: {overall_accuracy:.4f}")
        
        
        
    def process_data(self):
        """处理数据集并评估结果"""
        all_tasks = []
        for index, row in self.df.iterrows():
            if index not in self.test_qids or index in self.processed_tuples:
                continue
            all_tasks.append((index, row))
        
        print(f"开始处理 {len(all_tasks)} 条数据...")
        
        # 使用线程池处理任务
        with ThreadPoolExecutor(max_workers=self.args.num_threads) as executor:
            list(executor.map(self.process_single_row, all_tasks))
        
        # 评估不同阈值下的结果
        if self.args.thresholds:
            eval_results = self.evaluate_multiple_thresholds(self.args.thresholds)
        else:
            # 如果没有指定阈值，则评估无阈值情况
            eval_results = self.evaluate()
        
        return eval_results


def main():
    parser = argparse.ArgumentParser()
    # API相关参数
    parser.add_argument('--api_url', type=str, default="https://vip.yi-zhan.top/v1/chat/completions")
    parser.add_argument('--api_key', type=str, default="sk-pP1nu21e5A9Ucg9bB82c0dC07d334b9aA7A0Bc26F3Eb2f72")
    parser.add_argument('--model', type=str, default="gpt-4o-mini")
    parser.add_argument('--temperature', type=float, default=0.3)
    
    # 数据路径参数
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to education data directory')
    parser.add_argument('--retriever', type=str, required=True,
                      help='retriever name')
    parser.add_argument('--retrieval_results_path', type=str, required=True,
                      help='Path to retrieval results file')
    parser.add_argument('--collection_path', type=str, required=True,
                      help='Path to collection.tsv')
    parser.add_argument('--folds_path', type=str, required=True,
                      help='Path to folds.json')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to output jsonl file')
    
    # 其他参数
    parser.add_argument('--top_k', type=int, default=10,
                      help='Number of retrieved tables to use')
    parser.add_argument('--num_threads', type=int, default=4,
                      help='Number of threads for parallel processing')
    parser.add_argument('--thresholds', type=float, nargs='+',
                      help='List of confidence thresholds to evaluate')
    parser.add_argument('--evaluate_only', action='store_true',
                      help='只加载已有结果进行评估，不运行模型')

    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = str(Path(args.output_path).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    imputer = EducationImputer(args)
    
    if args.thresholds:
        for threshold in args.thresholds:
            imputer.evaluate_with_naive_rag_backup(threshold)
            


if __name__ == "__main__":
    main()