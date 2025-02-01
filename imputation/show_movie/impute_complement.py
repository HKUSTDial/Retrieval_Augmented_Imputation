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

class ShowMovieImputation:
    def __init__(self, 
                 model_name: str,
                 api_url: str,
                 api_key: str,
                 data_path: str = '/home/yangchenyu/Data_Imputation/data/show_movie',
                 output_path: str = '/home/yangchenyu/Data_Imputation/imputation/results/show_movie',
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
            output = output.replace('```json', '').replace('```', '')
            return ast.literal_eval(output)
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


    # def run_imputation(self, processed_tuples: list = None) -> float:
    #     """运行补全任务
    #     processed_tuples: 已处理过的tuple_ids列表，可选
    #     返回准确率
    #     """
    #     # 获取已处理的tuple_ids
    #     output_file = f'{self.output_path}/{self.model_name}_results_naive_RAG_{self.retriever}_top{self.top_k}.jsonl'
    #     processed_tuples = set()
        
    #     # 读取已有的结果文件，获取已处理的tuple_ids
    #     if os.path.exists(output_file):
    #         with jsonlines.open(output_file, 'r') as fin:
    #             for line in fin:
    #                 processed_tuples.add(line['tuple_id'])
    #         print(f"已找到{len(processed_tuples)}条处理过的记录")
            
    #     count, acc = 0, 0
        
    #     # 处理未完成的数据
    #     for index, row in self.df.iterrows():
    #         if index not in self.test_ids or index in processed_tuples:
    #             continue

    #         # 构建输入
    #         input_data = self.template + '[caption]: tv shows-movies\n'
            
    #         # 添加表头
    #         input_data += '|' + '|'.join(self.df.columns) + '|\n'
            
    #         # 添加示例
    #         input_data += '|' + '|'.join(str(x) for x in self.example) + '|\n'
            
    #         # 添加待补全数据
    #         row_data = []
    #         ground_truth = None
    #         for col in self.df.columns:
    #             if col == 'Rating':
    #                 cell_value = '[TO-FILL]'
    #                 ground_truth = row[col].lower().strip()
    #             else:
    #                 cell_value = row[col]
    #             row_data.append(str(cell_value))
            
    #         input_data += '|' + '|'.join(row_data) + '|'
    #         input_data = input_data.format(answer_format="{Rating: ''}")

    #         # 在构建输入时添加检索到的表格
    #         input_data += '\nRetrieved Tables:\n'
    #         retrieved_tables = self.topK_results[index]
    #         for rank, docid in enumerate(retrieved_tables):
    #             input_data += f'Table {rank+1}: {self.convert_to_table(self.collection[docid])}\n'

    #         print(f"\n处理第 {index} 条数据...")
    #         output = self.chat(input_data)
    #         imputed_data = self.extract_imputed_value(output)
            
    #         count += 1

    #         if imputed_data and 'Rating' in imputed_data:
    #             imputed_value = imputed_data['Rating'].lower().strip()
    #             if imputed_value in ground_truth:
    #                 acc += 1

    #             # 保存结果
    #             with jsonlines.open(output_file, 'a') as fout:
    #                 fout.write({
    #                     'tuple_id': index,
    #                     'input': input_data,
    #                     'output': output,
    #                     'ground_truth': ground_truth,
    #                     'prediction': imputed_value
    #                 })

    #     # 计算总体结果
    #     total_count, total_acc = 0, 0
    #     with jsonlines.open(output_file, 'r') as fin:
    #         for line in fin:
    #             total_count += 1
    #             if line['prediction'] in line['ground_truth']:
    #                 total_acc += 1

    #     accuracy = total_acc/total_count if total_count > 0 else 0
    #     print(f"\n当前批次准确率: {acc/count if count > 0 else 0:.4f} ({acc}/{count})")
    #     print(f"总体准确率: {accuracy:.4f} ({total_acc}/{total_count})")
    #     return accuracy


class ShowMovieImputer:
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
Query (Rating value does not have to be a numerical one!): 
{question}

Retrieved Tables:
{retrieved_tables}
        """
        
        self.url = args.api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}"
        }
        
        # 初始化results列表
        self.results = []  # 将results的初始化移到这里
        self.lock = threading.Lock()
        
        # 加载数据
        self.load_data()
        
        # 初始化naive RAG实例
        self.naive_rag = ShowMovieImputation(
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
        
        # 加载已处理的结果和tuple IDs
        self.processed_tuples = self.load_processed_tuples()
    
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
                    predictions[line['tuple_id']] = line
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
            # 匹配```json和```之间的内容
            json_pattern = r'```json\s*({[^`]*})\s*```'
            match = re.search(json_pattern, output, re.DOTALL)
            
            if match:
                json_str = match.group(1)
                try:
                    # 尝试解析JSON
                    result = json.loads(json_str)
                except json.JSONDecodeError:
                    try:
                        # 如果JSON解析失败，尝试用ast解析
                        result = ast.literal_eval(json_str)
                    except:
                        return None, None
            else:
                # 如果没有找到```json标记，尝试直接查找JSON对象
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
            if isinstance(result, dict) and 'Rating' in result:
                if isinstance(result['Rating'], dict):
                    value = result['Rating'].get('imputed_value', '').lower().strip()
                    confidence = float(result['Rating'].get('confidence', 0))
                    return value, confidence
                # 如果Rating直接是值而不是字典，则返回值和默认置信度1.0
                return str(result['Rating']).lower().strip(), 1.0
            return None, None
            
        except Exception as e:
            # print(f"处理输出时发生错误: {str(e)}")
            return None, None

    def process_single_row(self, task):
        """处理单行数据并记录原始结果"""
        index, row = task
        
        # 获取ground truth
        ground_truth = row['Rating'].lower().strip()
        
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
        # 构建question部分
        question = '[caption]: tv shows-movies\n'
        question += '|' + '|'.join(self.df.columns) + '|\n'
        
        # 构建数据行
        row_data = []
        for col in self.df.columns:
            if col == 'Rating':
                row_data.append('[TO-FILL]')
            else:
                row_data.append(str(row[col]))
        question += '|' + '|'.join(row_data) + '|\n'
        
        # 构建answer_format
        answer_format = '{"Rating": {"imputed_value": "", "confidence": ""}}'
        
        # 构建retrieved_tables部分
        retrieved_tables_text = ''
        if index in self.topK_results:
            for rank, docid in enumerate(self.topK_results[index]):
                retrieved_tables_text += f'Table {rank+1}: {self.convert_to_table(self.collection[docid])}\n\n'
        
        return self.template.format(
            answer_format=answer_format,
            question=question,
            retrieved_tables=retrieved_tables_text
        )
    
    def evaluate_with_naive_rag_backup(self, threshold=None, save_path=None):
        """使用naive RAG作为后备的评估方法"""
        if not self.existing_predictions:
            print("没有找到已有的预测结果！")
            return
        
        print(f"开始评估 {len(self.existing_predictions)} 条预测结果...")
        
        # 创建新的输出文件用于保存带有naive RAG补充的结果
        backup_output_path = self.args.output_path.replace('.jsonl', '_with_naive_rag_backup.jsonl')
        evaluated_results = []
        
        for tuple_id, result in self.existing_predictions.items():
            original_prediction = result['prediction']
            confidence = result.get('confidence')
            ground_truth = result['ground_truth']
            
            # 判断是否需要使用naive RAG
            need_backup = (threshold is not None and confidence is not None and confidence < threshold) or original_prediction is None
            
            if need_backup:
                naive_prediction = self.naive_rag.results[tuple_id]
                if naive_prediction and 'Rating' in naive_prediction:
                    final_prediction = naive_prediction['Rating'].lower().strip()
                else:
                    final_prediction = None
            else:
                final_prediction = original_prediction
            
            # 记录结果
            evaluated_result = {
                **result,
                'final_prediction': final_prediction,
                'used_naive_rag': need_backup,
                'is_correct': final_prediction in ground_truth if final_prediction else False
            }
            evaluated_results.append(evaluated_result)
            
            # 保存结果
            with jsonlines.open(backup_output_path, 'a') as fout:
                fout.write(evaluated_result)
        
        # 计算指标
        total = len(evaluated_results)
        rejected = sum(1 for r in evaluated_results if r['final_prediction'] is None)
        attempted = total - rejected
        correct = sum(1 for r in evaluated_results if r['is_correct'])
        naive_rag_used = sum(1 for r in evaluated_results if r['used_naive_rag'])
        naive_rag_correct = sum(1 for r in evaluated_results if r['used_naive_rag'] and r['is_correct'])
        
        # 计算各项指标
        accuracy = correct / attempted if attempted > 0 else 0
        coverage = attempted / total
        overall_accuracy = correct / total
        naive_rag_accuracy = naive_rag_correct / naive_rag_used if naive_rag_used > 0 else 0
        
        # 打印评估结果
        print(f"\n评估结果 (threshold={threshold}):")
        print(f"总样本数: {total}")
        print(f"拒绝预测数: {rejected}")
        print(f"尝试预测数: {attempted}")
        print(f"正确预测数: {correct}")
        print(f"准确率 (正确/尝试): {accuracy:.4f}")
        print(f"覆盖率 (尝试/总数): {coverage:.4f}")
        print(f"整体准确率 (正确/总数): {overall_accuracy:.4f}")
        print(f"使用naive RAG数: {naive_rag_used}")
        print(f"naive RAG正确数: {naive_rag_correct}")
        print(f"naive RAG准确率: {naive_rag_accuracy:.4f}")
        
        # 保存评估结果
        if save_path:
            eval_results = {
                'total_samples': total,
                'rejected_predictions': rejected,
                'attempted_predictions': attempted,
                'correct_predictions': correct,
                'accuracy': accuracy,
                'coverage': coverage,
                'overall_accuracy': overall_accuracy,
                'naive_rag_used': naive_rag_used,
                'naive_rag_correct': naive_rag_correct,
                'naive_rag_accuracy': naive_rag_accuracy,
                'confidence_threshold': threshold,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(save_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
        
        return eval_results


def main():
    parser = argparse.ArgumentParser()
    # API相关参数
    parser.add_argument('--api_url', type=str, default="https://vip.yi-zhan.top/v1/chat/completions")
    parser.add_argument('--api_key', type=str, default="OPENAI_API_KEY")
    parser.add_argument('--model', type=str, default="gpt-4o-mini")
    parser.add_argument('--temperature', type=float, default=0.3)
    
    # 数据路径参数
    parser.add_argument('--main_data_path', type=str, default="/home/yangchenyu/Data_Imputation/data/show_movie/tv_ret_1.csv",
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
    
    imputer = ShowMovieImputer(args)
    
    if args.thresholds:
        for threshold in args.thresholds:
            save_path = args.output_path.replace('.jsonl', f'_eval_with_naive_rag_t{threshold}.json')
            imputer.evaluate_with_naive_rag_backup(threshold, save_path)
    else:
        save_path = args.output_path.replace('.jsonl', '_eval_with_naive_rag.json')
        imputer.evaluate_with_naive_rag_backup(None, save_path)


if __name__ == "__main__":
    main()