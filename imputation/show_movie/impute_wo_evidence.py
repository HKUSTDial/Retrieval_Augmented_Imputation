import requests
import json
import jsonlines
import re
import time
import pandas as pd
import ast
from typing import Dict, Any, Optional
import os

import requests
import json
import jsonlines
import re
import time

url = "https://vip.yi-zhan.top/v1/chat/completions" 
openai_headers = { 
    "Content-Type": "application/json", 
    "Authorization": "Bearer OPENAI_API_KEY" 
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
                 output_path: str = '/home/yangchenyu/Data_Imputation/imputation/results/show_movie'):
        """
        初始化参数
        model_name: 使用的模型名称
        api_url: API接口地址
        api_key: API密钥
        data_path: 数据集路径
        output_path: 输出结果路径
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
            output = output.replace('```json', '').replace('```', '')
            return ast.literal_eval(output)
        except:
            return None

    def run_imputation(self) -> float:
        """运行补全任务,返回准确率"""
        count, acc = 0, 0
        output_file = f'{self.output_path}/{self.model_name}_results_wo_evid.jsonl'
        
        # 读取已处理的数据
        processed_data = {}
        if os.path.exists(output_file):
            with jsonlines.open(output_file, 'r') as fin:
                for line in fin:
                    processed_data[line['tuple_id']] = line

        for index, row in self.df.iterrows():
            if index not in self.test_ids:
                continue
            
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
            

            if index in processed_data:
                # 从processed_data中获取已有的预测结果
                output = processed_data[index]['output']
                imputed_data = self.extract_imputed_value(output)
            else:
                # 构建输入
                input_data = self.template + '[caption]: tv shows-movies\n'
                
                # 添加表头
                input_data += '|' + '|'.join(self.df.columns) + '|\n'
                
                # 添加示例
                input_data += '|' + '|'.join(str(x) for x in self.example) + '|\n'
                
                
                input_data += '|' + '|'.join(row_data) + '|'
                input_data = input_data.format(answer_format="{Rating: ''}")

                print(f"\n处理第 {index} 条数据...")
                output = self.chat(input_data)
                imputed_data = self.extract_imputed_value(output)
                
                # 保存结果
                with jsonlines.open(output_file, 'a') as fout:
                    fout.write({
                        'tuple_id': index,
                        'input': input_data,
                        'output': output,
                        'ground_truth': ground_truth,
                        'prediction': imputed_data['Rating'] if imputed_data else ''
                    })
            
            if imputed_data and 'Rating' in imputed_data:
                imputed_value = imputed_data['Rating'].lower().strip()
                if imputed_value in ground_truth:
                    acc += 1
                count += 1

        accuracy = acc/count if count > 0 else 0
        print(f"\n最终准确率: {accuracy:.4f} ({acc}/{count})")
        return accuracy



# 初始化并运行
imputer = ShowMovieImputation(
    model_name="gpt-4o-mini",
    api_url="https://vip.yi-zhan.top/v1/chat/completions",
    api_key="OPENAI_API_KEY",
)

# 运行补全任务
accuracy = imputer.run_imputation()