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

class EducationImputer:
    def __init__(self, args):
        self.args = args
        self.template = """## Task
Assess whether missing values in the query table can be imputed using retrieved tables, ensuring the data is sufficient, relevant, and logically consistent:

1. **Examine the Input**: Review the query table with missing value and the provided "Retrieved Tables."
2. **Evaluate Retrieved Tables**: Use the following dimensions to assess the usefulness of each retrieved table for filling missing values in the query table:
* Existence: Does the tuple include at least one missing attribute? (Yes/No)
* Relevance: How related is the tuple to the query? (Highly Relevant/Somewhat Relevant/Not Relevant)
* Logical Consistency: Is the tuple logically aligned with the query? (Fully Consistent/Partially Consistent/Not Consistent)

3. **Generate Response**: 
- Based on the above analysis, if sufficient information exists, respond using JSON format: {answer_format} with confidence score (0-1) for each imputed attribute. The confidence level for each imputation is based on the reliability of the retrieved tuples and the rigor of the reasoning used to fill the missing values. 
- If imputation is not feasible, return: "Sorry I can't provide the grounded imputed values since retrieved data do not contain useful information for imputation." 


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
        
        self.results = []
        self.lock = threading.Lock()
        
        # 加载数据
        self.load_data()
        
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
        
        # 设置要补全的列
        self.missing_columns = ['Street Address', 'ZIP Code', 'Phone Number']
        
        # 加载已处理的结果
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
                all_scores[int(qid)][int(docid)] = score
        
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

    def process_single_row(self, qid):
        """处理单行数据并记录原始结果"""
        # 构建输入
        input_data = self.construct_input(qid)
        
        # 获取输出
        output = self.chat(input_data)
        
        # 提取预测值和置信度
        prediction, confidence = self.extract_prediction(output)
        
        # 记录原始结果（不应用阈值）
        result = {
            'tuple_id': qid,
            'input': input_data,
            'output': output,
            'ground_truth': self.ground_truth[qid],
            'prediction': prediction,
            'confidence': confidence,
        }
        
        # 安全写入文件和结果列表
        with self.lock:
            self.results.append(result)
            with jsonlines.open(self.args.output_path, 'a') as fout:
                fout.write(result)

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

    def evaluate(self, threshold=None):
        """评估结果"""
        total = len(self.results)
        if total == 0:
            print("没有处理任何数据！")
            return
        
        evaluated_results = []
        acc_by_column = defaultdict(int)
        attempted_by_column = defaultdict(int)
        rejected_by_column = defaultdict(int)
        total_by_column = defaultdict(int)
        
        # 添加整体指标的计数器
        total_all = 0
        attempted_all = 0
        rejected_all = 0
        correct_all = 0
        
        for r in self.results:
            predictions = r.get('prediction', {}) or {}
            confidences = r.get('confidence', {}) or {}
            ground_truths = r.get('ground_truth', {}) or {}
            
            result = {'tuple_id': r['tuple_id']}
            
            for col in self.missing_columns:
                # 只处理有ground truth的列
                if col in ground_truths and ground_truths[col]:
                    total_by_column[col] += 1
                    total_all += 1
                    
                    if not predictions or col not in predictions:
                        rejected_by_column[col] += 1
                        rejected_all += 1
                        continue
                    
                    if confidences[col] is not None and confidences[col] < threshold:
                        rejected_by_column[col] += 1
                        rejected_all += 1
                        continue

                    attempted_all += 1
                        
                    prediction = predictions.get(col)
                    confidence = confidences.get(col, 0)
                    
                    # 应用阈值
                    if threshold is not None and confidence < threshold:
                        rejected_by_column[col] += 1
                        rejected_all += 1
                        continue
                    # print(prediction, ground_truths[col])
                    if type(ground_truths) == str:
                        if prediction.lower().strip() in ground_truths[col].lower().strip():
                            acc_by_column[col] += 1
                            correct_all += 1
                    elif type(ground_truths[col]) == list:
                        if any(prediction.lower().strip() in gold_answer.lower().strip() for gold_answer in ground_truths[col]):
                            acc_by_column[col] += 1
                            correct_all += 1
                    else:
                        rejected_by_column[col] += 1
                        rejected_all += 1
            
            evaluated_results.append(result)
        
        # 计算整体指标
        overall_accuracy = correct_all / attempted_all if attempted_all > 0 else 0
        overall_coverage = attempted_all / total_all if total_all > 0 else 0
        overall_total_accuracy = correct_all / total_all if total_all > 0 else 0
        
        # 打印整体评估结果
        print(f"\n整体评估结果 (threshold={threshold}):")
        print(f"总预测次数: {total_all}")
        print(f"拒绝预测数: {rejected_all}")
        print(f"尝试预测数: {attempted_all}")
        print(f"正确预测数: {correct_all}")
        print(f"准确率 (正确/尝试): {overall_accuracy:.4f}")
        print(f"覆盖率 (尝试/总数): {overall_coverage:.4f}")
        print(f"整体准确率 (正确/总数): {overall_total_accuracy:.4f}")
        
        
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

    def process_data(self):
        """处理数据集并评估结果"""
        all_tasks = []
        for qid in self.test_qids:
            if qid in self.processed_tuples:
                continue
            all_tasks.append(qid)
        
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
    parser.add_argument('--api_key', type=str, default="OPENAI_API_KEY")
    parser.add_argument('--model', type=str, default="gpt-4o-mini")
    parser.add_argument('--temperature', type=float, default=0.3)
    
    # 数据路径参数
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to education data directory')
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
    
    imputer = EducationImputer(args)
    
    imputer.process_data()
    

if __name__ == "__main__":
    main()

