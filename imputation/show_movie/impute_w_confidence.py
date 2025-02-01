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

class ShowMovieImputer:
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
- Based on the above analysis, if sufficient information exists: respond using JSON format: {answer_format} with confidence score （0-1）for each imputed attribute. The confidence level for each imputation is based on the reliability of the retrieved tuples and the rigor of the reasoning used to fill the missing values.
- If imputation is not feasible, return: "Sorry I can't provide the grounded imputed values since retrieved data do not contain useful information for imputation." 

Note: Only use information grounded in the retrieved tables. Do not rely on internal knowledge or unsupported reasoning.

## Input
Query (Rating does not have to be a numerical value): 
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
        
    # def load_retrieval_results(self):
    #     """加载检索结果"""
    #     topK_results = defaultdict(list)
    #     with open(self.args.retrieval_results_path, 'r') as f:
    #         for line in f:
    #             qid, docid, rank, score = line.strip().split('\t')
    #             if int(rank) < self.args.top_k:
    #                 topK_results[int(qid)].append(int(docid))
    #     return topK_results
    
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

    def evaluate(self, threshold=None, save_path=None):
        """使用指定阈值评估结果
        
        Args:
            threshold: 置信度阈值，如果为None则不使用阈值
            save_path: 评估结果保存路径，如果为None则使用默认路径
        """
        total = len(self.results)
        if total == 0:
            print("没有处理任何数据！")
            return
        
        # 应用阈值进行评估
        evaluated_results = []
        for r in self.results:
            prediction = r['prediction']
            confidence = r['confidence']
            
            # 如果设置了阈值且置信度低于阈值，拒绝该预测
            if threshold is not None and confidence is not None and confidence < threshold:
                prediction = None
                
            evaluated_results.append({
                **r,
                'thresholded_prediction': prediction,
                'is_rejected': prediction is None,
                'is_correct': prediction in r['ground_truth'] if prediction else False
            })
        
        # 计算指标
        rejected = sum(1 for r in evaluated_results if r['is_rejected'])
        attempted = total - rejected
        correct = sum(1 for r in evaluated_results if r['is_correct'])
        
        accuracy = correct / attempted if attempted > 0 else 0
        coverage = attempted / total
        overall_accuracy = correct / total
        
        # 准备评估结果
        eval_results = {
            'total_samples': total,
            'rejected_predictions': rejected,
            'attempted_predictions': attempted,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'coverage': coverage,
            'overall_accuracy': overall_accuracy,
            'confidence_threshold': threshold,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 打印评估结果
        print(f"\n评估结果 (threshold={threshold}):")
        print(f"总样本数: {total}")
        print(f"拒绝预测数: {rejected}")
        print(f"尝试预测数: {attempted}")
        print(f"正确预测数: {correct}")
        print(f"准确率 (正确/尝试): {accuracy:.4f}")
        print(f"覆盖率 (尝试/总数): {coverage:.4f}")
        print(f"整体准确率 (正确/总数): {overall_accuracy:.4f}")
        
        # 保存评估结果
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
        
        return eval_results

    def evaluate_multiple_thresholds(self, thresholds):
        """评估多个阈值下的性能
        
        Args:
            thresholds: 要评估的阈值列表
        """
        results = {}
        for threshold in thresholds:
            print(f"\n评估阈值 {threshold}...")
            save_path = self.args.output_path.replace('.jsonl', f'_eval_t{threshold}.json')
            results[threshold] = self.evaluate(threshold, save_path)
        
        # 保存所有阈值的结果
        all_results_path = self.args.output_path.replace('.jsonl', '_all_thresholds_eval.json')
        with open(all_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

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
    
    # 如果不是只评估模式，则运行补全任务
    if not args.evaluate_only:
        imputer.process_data()
    
    # 如果指定了阈值，评估不同阈值下的结果
    if args.thresholds:
        eval_results = imputer.evaluate_multiple_thresholds(args.thresholds)
    else:
        # 如果没有指定阈值，则评估无阈值情况
        eval_results = imputer.evaluate()

if __name__ == "__main__":
    main()