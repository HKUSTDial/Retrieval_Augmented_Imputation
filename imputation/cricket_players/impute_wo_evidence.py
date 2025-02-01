import requests
import json
import jsonlines
import re
import time
import pandas as pd
import ast
import os
from typing import Dict, Any, Optional
from collections import defaultdict

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

class CricketPlayersImputation:
    def __init__(self, 
                 model_name: str,
                 api_url: str,
                 api_key: str,
                 data_path: str = '/home/yangchenyu/Data_Imputation/data/cricket_players',
                 output_path: str = '/home/yangchenyu/Data_Imputation/imputation/results/cricket_players'):
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
        
        # 加载数据
        self.keep_columns = ['Name', 'Team', 'Type', 'ValueinCR', 'National Side', 
                           'Batting Style', 'Bowling', 'MatchPlayed', 'InningsBatted', 'RunsScored']
        self.df = pd.read_csv(f'{data_path}/cricket_ret_1.csv', usecols=self.keep_columns)
        
        with open(f'{data_path}/annotated_data/folds.json', 'r') as f:
            self.test_ids = json.load(f)['test']
            
        # 设置要补全的列
        self.missing_columns = ['National Side', 'Batting Style']
        
        
        # 模板
        self.template = '''What's the most likely value for the [TO-FILL] cell in the table below? Please respond using JSON: {answer_format}, the key is attribute name of each [TO-FILL], value is the predicted value for each [TO-FILL].\n'''

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

    def run_imputation(self, processed_tuples: list = None) -> float:
        """运行补全任务"""
            
        count, acc = 0, 0
        acc_by_column = defaultdict(int)
        output_file = f'{self.output_path}/{self.model_name}_wo_evidence_results.jsonl'
        
        processed_data = {}
        if os.path.exists(output_file):
            with jsonlines.open(output_file, 'r') as fin:
                for line in fin:
                    processed_data[line['tuple_id']] = line

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

            if index in processed_data:
                output = processed_data[index]['output']
                imputed_data = self.extract_imputed_value(output)
            else:

                input_data = self.template + '[caption]: cricket player\n'
                
                # 添加表头
                input_data += '|' + '|'.join(self.df.columns) + '|\n'
                
                answer_format = answer_format[:-2] + '}'
                input_data += '|' + '|'.join(row_data) + '|'
                input_data = input_data.format(answer_format=answer_format)

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
                        true_value = ground_truths[col]
                        if col == 'Batting Style':
                            print(imputed_value, true_value)
                            if 'right' in imputed_value and 'right' in true_value:
                                acc_by_column[col] += 1
                                correct_predictions += 1
                            elif 'left' in imputed_value and 'left' in true_value:
                                acc_by_column[col] += 1
                                correct_predictions += 1
                        else:
                            if imputed_value == true_value:
                                acc_by_column[col] += 1
                                correct_predictions += 1

                

            acc += correct_predictions / len(self.missing_columns)

        # 计算总体结果
        accuracy = acc / count if count > 0 else 0
        print(f"\n总体准确率: {accuracy:.4f} ({acc}/{count})")
        
        # 打印每个列的准确率
        for col in self.missing_columns:
            col_accuracy = acc_by_column[col] / count if count > 0 else 0
            print(f"{col} 准确率: {col_accuracy:.4f} ({acc_by_column[col]}/{count})")
            
        return accuracy

    def evaluate_from_file(self, result_file: str) -> float:
        """从已有的结果文件中评估准确率"""
        count = 0
        acc = 0
        acc_by_column = defaultdict(int)
        
        print(f"从文件加载结果: {result_file}")
        
        with jsonlines.open(result_file, 'r') as fin:
            for line in fin:
                ground_truth = line.get('ground_truth', {})
                prediction = line.get('prediction', {})
                
                if not ground_truth or not prediction:
                    continue
                
                count += 1
                correct_predictions = 0
                
                for col in self.missing_columns:
                    if col in prediction and col in ground_truth:
                        pred_value = prediction[col].lower().strip()
                        true_value = ground_truth[col].lower().strip()
                        if pred_value == true_value:
                            acc_by_column[col] += 1
                            correct_predictions += 1
                
                acc += correct_predictions / len(self.missing_columns)
        
        accuracy = acc / count if count > 0 else 0
        print(f"\n总体准确率: {accuracy:.4f} ({acc}/{count})")
        
        for col in self.missing_columns:
            col_accuracy = acc_by_column[col] / count if count > 0 else 0
            print(f"{col} 准确率: {col_accuracy:.4f} ({acc_by_column[col]}/{count})")
        
        return accuracy

# 使用示例
if __name__ == "__main__":
    imputer = CricketPlayersImputation(
        model_name="gpt-4o",
        api_url="https://vip.yi-zhan.top/v1/chat/completions",
        api_key="sk-pP1nu21e5A9Ucg9bB82c0dC07d334b9aA7A0Bc26F3Eb2f72",
    )

    # 运行补全任务
    accuracy = imputer.run_imputation()

    # 或者从已有结果文件评估
    # result_file = "./results/Cricket_Players/gpt4_results.jsonl"
    # accuracy = imputer.evaluate_from_file(result_file)