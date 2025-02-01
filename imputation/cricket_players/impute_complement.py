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

class CricketPlayersImputation:
    def __init__(self, 
                 model_name: str,
                 api_url: str,
                 api_key: str,
                 data_path: str = '/home/yangchenyu/Data_Imputation/data/cricket_players',
                 output_path: str = '/home/yangchenyu/Data_Imputation/imputation/results/cricket_players',
                 retrieval_file: str = None,
                 retriever: str = 'BM25',
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
        self.retriever = retriever
        
        # 加载数据
        self.keep_columns = ['Name', 'Team', 'Type', 'ValueinCR', 'National Side', 
                           'Batting Style', 'Bowling', 'MatchPlayed', 'InningsBatted', 'RunsScored']
        self.df = pd.read_csv(f'{data_path}/cricket_ret_1.csv', usecols=self.keep_columns)
        
        with open(f'{data_path}/annotated_data/folds.json', 'r') as f:
            self.test_ids = json.load(f)['test']
        
        
        # 设置检索文件路径
        self.top_k = top_k
        self.retrieval_file = retrieval_file or f'{self.data_path}/rerank_results/cricket_players_test.tsv'
        
        # 加载检索结果和集合数据
        self.topK_results = defaultdict(list)
        self.collection = {}
        self._load_retrieval_results()
        self._load_collection()
        self.load_processed_tuples()
        
        # 更新模板
        self.template = '''Based on the retrieved tabular data, what's the most likely value for the [TO-FILL] cell in the table below? Please respond using JSON: {answer_format}, the key is attribute name of each [TO-FILL], value is the predicted value for each [TO-FILL].\n'''

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


class CricketPlayersImputer:
    def __init__(self, args):
        self.args = args
        self.template = """## Task
Assess whether missing values in the query table can be imputed using retrieved tables, ensuring the data is sufficient, relevant, and logically consistent:

1. **Examine the Input**: Review the query table with missing value and the provided "Retrieved Tables."
2. **Evaluate Retrieved Tables**: Use the following dimensions to assess the usefulness of each retrieved table for filling missing values in the query table:
* Existence: Does the tuple include at least one attribute to be filled? (Yes/No)
* Relevance: How related is the tuple to the query? (Highly Relevant/Somewhat Relevant/Not Relevant)
* Logical Consistency: Is the tuple logically aligned with the query (e.g., temporal consistency, dependencies)? (Fully Consistent/Partially Consistent/Not Consistent)


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
        self.missing_columns = ['National Side', 'Batting Style']
        # 初始化results列表
        self.results = []  # 将results的初始化移到这里
        self.lock = threading.Lock()
        
        # 加载数据
        self.load_data()
        
        # 初始化naive RAG实例
        self.naive_rag = CricketPlayersImputation(
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
        
        # 加载主数据集
        self.df = pd.read_csv(self.args.main_data_path)
        
    
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

    def process_single_row(self, task):
        """处理单行数据并记录原始结果"""
        index, row = task
        
        # 获取ground truth，跳过原本就是空值的情况
        ground_truth = {}
        has_valid_ground_truth = False
        for col in self.missing_columns:
            if pd.notna(row[col]) and isinstance(row[col], str):  # 确保值不是空且是字符串
                ground_truth[col] = row[col].lower().strip()
                has_valid_ground_truth = True
        
        # 如果没有有效的ground truth，跳过这个样本
        if not has_valid_ground_truth:
            print(f"跳过样本 {index}: 没有有效的ground truth")
            return
        
        # 构建输入
        input_data = self.construct_input(row, index)
        
        # 获取输出
        output = self.chat(input_data)
        
        # 提取预测值和置信度
        prediction, confidence = self.extract_prediction(output)
        
        # 记录原始结果（不应用阈值）
        result = {
            'tuple_id': index,
            'input': input_data,
            'output': output,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'confidence': confidence,
        }
        
        # 安全写入文件和结果列表
        with self.lock:
            self.results.append(result)
            with jsonlines.open(self.args.output_path, 'a') as fout:
                fout.write(result)

    def construct_input(self, row, index):
        """构建模型输入"""
        question = '[caption]: cricket player\n'
        question += '|' + '|'.join(self.df.columns) + '|\n'
        
        row_data = []
        for col in self.df.columns:
            if col in self.missing_columns:
                row_data.append('[TO-FILL]')
            else:
                row_data.append(str(row[col]))
        question += '|' + '|'.join(row_data) + '|\n'
        
        # 构建answer_format
        answer_format = '{'
        for col in self.missing_columns:
            answer_format += f'"{col}": {{"imputed_value": "", "confidence": ""}}, '
        answer_format = answer_format[:-2] + '}'
        
        retrieved_tables_text = ''
        if index in self.topK_results:
            for rank, docid in enumerate(self.topK_results[index]):
                retrieved_tables_text += f'Table {rank+1}: {self.convert_to_table(self.collection[docid])}\n\n'
        
        return self.template.format(
            answer_format=answer_format,
            question=question,
            retrieved_tables=retrieved_tables_text
        )

    def evaluate(self, threshold=None):
        """使用指定阈值评估结果，对于拒绝的预测使用naive RAG策略"""
        total = len(self.results)
        if total == 0:
            print("没有处理任何数据！")
            return
        
        # 初始化指标
        evaluated_results = []
        acc_by_column = defaultdict(int)
        attempted_by_column = defaultdict(int)
        rejected_by_column = defaultdict(int)
        total_by_column = defaultdict(int)
        naive_rag_used_by_column = defaultdict(int)
        naive_rag_correct_by_column = defaultdict(int)
        
        for r in self.results:
            predictions = r.get('prediction', {}) or {}
            confidences = r.get('confidence', {}) or {}
            ground_truths = r.get('ground_truth', {}) or {}
            
            result = {'tuple_id': r['tuple_id']}
            
            for col in self.missing_columns:
                # 只处理有ground truth的列
                if col in ground_truths:
                    total_by_column[col] += 1
                    
                    prediction = predictions.get(col)
                    confidence = confidences.get(col)
                    
                    # 如果设置了阈值且置信度低于阈值，或预测为None，使用naive RAG
                    if (threshold is not None and (confidence is None or confidence < threshold)) or prediction is None:
                        rejected_by_column[col] += 1
                        
                        # 获取naive RAG的预测
                        naive_prediction = self.naive_rag.results[r['tuple_id']]
                        
                        if naive_prediction and col in naive_prediction:
                            prediction = naive_prediction[col].lower().strip()
                            naive_rag_used_by_column[col] += 1
                            
                            if col == 'Batting Style':
                                if ('right' in prediction and 'right' in ground_truths[col]) or \
                                   ('left' in prediction and 'left' in ground_truths[col]):
                                    naive_rag_correct_by_column[col] += 1
                            else:
                                if prediction == ground_truths[col]:
                                    naive_rag_correct_by_column[col] += 1
                            
                        result[col] = {
                            'ground_truth': ground_truths[col],
                            'prediction': prediction,
                            'confidence': confidence,
                            'used_naive_rag': prediction != predictions.get(col)
                        }
            
            evaluated_results.append(result)
        
        # 计算每列指标
        results_by_column = {}
        for col in self.missing_columns:
            total = total_by_column[col]
            if total == 0:
                continue
            
            attempted = attempted_by_column[col] 
            rejected = rejected_by_column[col]
            correct = acc_by_column[col]
            naive_rag_used = naive_rag_used_by_column[col]
            naive_rag_correct = naive_rag_correct_by_column[col]
            
            accuracy = correct / attempted if attempted > 0 else 0
            coverage = attempted / total
            overall_accuracy = correct / total
            naive_rag_accuracy = naive_rag_correct / naive_rag_used if naive_rag_used > 0 else 0
            
            results_by_column[col] = {
                'total_samples': total,
                'rejected_predictions': rejected, 
                'attempted_predictions': attempted,
                'correct_predictions': correct,
                'accuracy': accuracy,
                'coverage': coverage,
                'overall_accuracy': overall_accuracy,
                'naive_rag_used': naive_rag_used,
                'naive_rag_correct': naive_rag_correct,
                'naive_rag_accuracy': naive_rag_accuracy
            }
        
        # 打印每列评估结果
        print(f"\n评估结果 (threshold={threshold}):")
        for col, metrics in results_by_column.items():
            print(f"\n{col}列:")
            for metric, value in metrics.items():
                if 'accuracy' in metric:
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")
        
        return evaluated_results

    def evaluate_multiple_thresholds(self, thresholds):
        """评估多个阈值下的性能
        
        Args:
            thresholds: 要评估的阈值列表
        """
        results = {}
        for threshold in thresholds:
            print(f"\n评估阈值 {threshold}...")
            results[threshold] = self.evaluate(threshold)
        
        # 保存所有阈值的结果
        all_results_path = self.args.output_path.replace('.jsonl', '_all_thresholds_eval.json')
        with open(all_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_with_naive_rag_backup(self, threshold=None, save_path=None):
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
                # print(attr, ground_truths)
                if attr in ground_truths:
                    total_samples += 1

                    try:
                        original_prediction = original_predictions.get(attr)
                        confidence = confidences.get(attr)
                    except:
                        original_prediction = None
                        confidence = None
                    
                    # 判断是否需要使用naive RAG
                    need_backup = (threshold is not None and confidence is not None and confidence < threshold) or original_prediction is None

                    filled_count += 1
                    
                    if need_backup:
                        if tuple_id not in self.naive_rag.results:
                            print(f"tuple_id {tuple_id} 不在naive RAG结果中")
                            continue
                        naive_prediction = self.naive_rag.results[tuple_id]
                        
                        if naive_prediction and attr in naive_prediction:
                            final_prediction = naive_prediction[attr].lower().strip()
                            
                            if attr == 'Batting Style':
                                if ('right' in final_prediction and 'right' in ground_truths[attr]) or \
                                   ('left' in final_prediction and 'left' in ground_truths[attr]):
                                    is_correct[attr] = True
                                    total_correct += 1
                                else:
                                    is_correct[attr] = False
                            else:
                                is_correct[attr] = final_prediction == ground_truths[attr]
                                if is_correct[attr]:
                                    total_correct += 1
                        else:
                            final_prediction = None
                            is_correct[attr] = False
                            
                        used_naive_rag[attr] = True
                            
                    else:
                        final_prediction = original_prediction
                        
                        if attr == 'Batting Style':
                            if ('right' in final_prediction and 'right' in ground_truths[attr]) or \
                               ('left' in final_prediction and 'left' in ground_truths[attr]):
                                is_correct[attr] = True
                                total_correct += 1
                            else:
                                is_correct[attr] = False
                        else:
                            is_correct[attr] = final_prediction == ground_truths[attr]
                            if is_correct[attr]:
                                total_correct += 1
                                
                        used_naive_rag[attr] = False
                        
                    final_predictions[attr] = final_prediction
            
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
    parser.add_argument('--main_data_path', type=str, default="/home/yangchenyu/Data_Imputation/data/cricket_players/cricket_ret_1.csv",
                      help='Path to tv_ret_1.csv')
    parser.add_argument('--retrieval_results_path', type=str, required=True,
                      help='Path to retrieval results file')
    parser.add_argument('--collection_path', type=str, required=True,
                      help='Path to collection.tsv')
    parser.add_argument('--folds_path', type=str, required=True,
                      help='Path to folds.json')
    parser.add_argument('--retriever', type=str, required=True,
                      help='name of retriever')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to output jsonl file')
    
    # 其他参数
    parser.add_argument('--top_k', type=int, default=10,
                      help='Number of retrieved tables to use')
    parser.add_argument('--num_threads', type=int, default=4,
                      help='Number of threads for parallel processing')
    
    # 修改threshold参数为可选的多值参数
    parser.add_argument('--thresholds', type=float, nargs='+',
                      help='List of confidence thresholds to evaluate')
    
    # 添加只进行评估的参数
    parser.add_argument('--evaluate_only', action='store_true',
                      help='只加载已有结果进行评估，不运行模型')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = str(Path(args.output_path).parent)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    imputer = CricketPlayersImputer(args)
    
    if args.thresholds:
        for threshold in args.thresholds:
            save_path = args.output_path.replace('.jsonl', f'_eval_with_naive_rag_t{threshold}.json')
            imputer.evaluate_with_naive_rag_backup(threshold, save_path)
    else:
        save_path = args.output_path.replace('.jsonl', '_eval_with_naive_rag.json')
        imputer.evaluate_with_naive_rag_backup(None, save_path)


if __name__ == "__main__":
    main()