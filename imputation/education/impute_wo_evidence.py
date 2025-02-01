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
                ):
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
        
        
        self.collection = {}
        self._load_collection()
        
        # 更新模板
        self.template = '''What's the most likely value for the [TO-FILL] cell in the table below? Please respond using JSON: {'Street Address': '', 'ZIP Code': '', 'Phone Number': ''}, the key is attribute name of each [TO-FILL], value is the predicted value for each [TO-FILL].\n'''
        
        # 添加标注文件路径
        self.annotation_file = f'{self.output_path}/{self.model_name}_wo_evidence/manual_annotations.jsonl'
        if not os.path.exists(self.annotation_file):
            os.makedirs(os.path.dirname(self.annotation_file), exist_ok=True)
        self.existing_annotations = self._load_annotations(self.annotation_file)
    

    def chat(self, input_data, model="gpt-4o-mini", temperate=0.8):
        
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


    def _load_annotations(self, annotation_file):
        """加载已有的标注结果"""
        annotations = {}
        if os.path.exists(annotation_file):
            with jsonlines.open(annotation_file, 'r') as f:
                for line in f:
                    key = (line['tuple_id'], line['header'], line['prediction'])
                    annotations[key] = line
        return annotations
     
    
    def save_manual_annotation(self, annotation_file, annotation):
        """保存人工标注结果"""
        with open(annotation_file, 'a', encoding='utf-8') as f:
            json.dump(annotation, f)
            f.write('\n')

    
    def verify_entity_equivalence(self, prediction, answer_set, t_id, hint):
        """验证预测值和答案集是否表示相同实体"""
        annotation_key = (t_id, hint, prediction)
        if annotation_key in self.existing_annotations:
            existing_annotation = self.existing_annotations[annotation_key]
            # print(f"使用已有标注 - Tuple ID: {t_id}, Header: {hint}, 结果: {'正确' if existing_annotation['manual_check_result'] else '错误'}")
            return existing_annotation['manual_check_result']
        if len(answer_set) > 30:
            answer_set = answer_set[:30]
            
        prompt = f"""Strictly determine if the prediction and any item in the answer set refer to EXACTLY the same entity/concept. They should describe the identical entity with only minor textual variations (e.g., formatting, capitalization, spacing).

    Examples of equivalent pairs:
    - "New York City" = "NYC"
    - "United States of America" = "USA"
    - "William Shakespeare" = "Shakespeare, William"

    Examples of non-equivalent pairs:
    - "New York City" ≠ "New York State"
    - "Apple Inc." ≠ "Apple Computer"
    - "John F. Kennedy" ≠ "John Kennedy Jr."

    Prediction: {prediction}
    Answer Set: {answer_set}

    Answer only 'Yes' or 'No'."""

        try:
            response = self.chat(prompt, model="gpt-4o-mini", temperate=0.1)
            is_correct = any(word in response.lower() for word in ['yes', 'y'])
            # 保存LLM的标注结果
            annotation = {
                'tuple_id': t_id,
                'header': hint,
                'prediction': prediction,
                'ground_truth': list(answer_set),
                'manual_check_result': is_correct,
                'annotation_type': 'llm',  # 标记这是LLM的标注
                'llm_response': response,  # 保存LLM的原始响应
            }
            self.save_manual_annotation(self.annotation_file, annotation)
            
            # 更新已有标注字典
            self.existing_annotations[annotation_key] = annotation
            
            return is_correct
        except Exception as e:
            print(f"LLM调用出错: {str(e)}")
            return False
    

    def run_imputation(self) -> float:
        """运行补全任务"""
        output_file = f'{self.output_path}/{self.model_name}_wo_evidence_results.jsonl'
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

                input_data = self.template + query 

                print(input_data)

                print(f"\n处理第 {qid} 条数据...")
                output = self.chat(input_data)
                imputed_data = self.extract_imputed_value(output)
                print(output)
                print(imputed_data)

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
                            if self.verify_entity_equivalence(
                                imputed_data[col],
                                set(correct_values[col]),
                                qid,
                                col
                            ):
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


    def extract_results_from_file(self, result_file: str) -> pd.DataFrame:
        """从结果文件中提取预测值并生成DataFrame"""
        results = []
        
        with jsonlines.open(result_file, 'r') as fin:
            for line in fin:
                tuple_id = line['tuple_id']
                prediction = line.get('prediction', {})
                
                result = {'tuple_id': tuple_id}
                
                # 添加预测值
                for col in self.missing_columns:
                    result[f'{col}_pred'] = prediction.get(col, '')
                
                results.append(result)
        
        return pd.DataFrame(results)


imputer = EducationImputation(
    model_name="gpt-4o",
    api_url="https://vip.yi-zhan.top/v1/chat/completions",
    api_key="sk-pP1nu21e5A9Ucg9bB82c0dC07d334b9aA7A0Bc26F3Eb2f72",
)

# 运行补全任务
accuracy = imputer.run_imputation()


# # 提取结果到DataFrame进行进一步分析
# results_df = imputer.extract_results_from_file(result_file)
# print(results_df.head())