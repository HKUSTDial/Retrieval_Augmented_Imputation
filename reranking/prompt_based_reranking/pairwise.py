import requests
import json
import re
import time
import pandas as pd
from collections import defaultdict
import openai


openai.api_key = "OPENAI_API_KEY"
openai.api_base = "OPENAI_API_BASE"



class ReRanker():
    def __init__(self, path, samples, window_size, startq):
        self.data = pd.read_csv(path)
        
        self.doc_ids = self.data['docid'].to_list()
        
        # 检查数据列是否存在
        if 'caption_d' in self.data.columns and 'content_d' in self.data.columns:
            self.doc_tuples = (self.data['caption_d'] + self.data['content_d']).to_list()
        else:
            # 假设只有content_d列
            self.doc_tuples = self.data['content_d'].to_list()
        
        self.doc_dict = defaultdict(str)
        for i, d in enumerate(self.doc_ids):
            self.doc_dict[d] = self.doc_tuples[i]

        self.q_ids = self.data['qid'].to_list()
        
        # 同样检查查询相关的列
        if 'caption_q' in self.data.columns and 'content_q' in self.data.columns:
            self.q_tuples = (self.data['caption_q'] + self.data['content_q']).to_list()
        else:
            # 假设只有content_q列
            self.q_tuples = self.data['content_q'].to_list()
        
        self.q_num = len(self.q_ids) // 100
        self.q_dict = defaultdict(str)
        for i, q in enumerate(self.q_ids):
            self.q_dict[q] = self.q_tuples[i]
    

        self.samples = samples
        self.window_size = window_size
        self.step_size = 1
        self.round = 1

        self.startq = startq

    def chat(self, input_data, model="gpt-4o-mini"):

        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": input_data}],
                temperature=0.3,
            )
            output_content = response.choices[0].message.content
            print(output_content)
            return output_content
        
        except Exception as e:
            print(f"API 调用出错: {e}")
            time.sleep(1)
            return self.chat(input_data, model)  # 递归重试
            
            
    def getprompt_rankgpt(self, q_tuple, doc_ids):
        prompt = []
        prompt.append(
            'I will provide you with 2 passages, each indicated by number identifier [identifier]  and a query.')

        for i in range(self.window_size):
            prompt.append('Passage [' + str(doc_ids[i]) + '] = {' + str(self.doc_dict[doc_ids[i]]) + '}.')
        prompt.append('Query = {' + q_tuple + '}.')
        prompt.append(
            'Which passage is more likely to provide relevant information to infer the missing value N/A of the Query serving as a reference for the missing value? Only return the identifier, do not say any word or explain.')
        return '\n'.join(prompt)

    def pairwise_rank(self):
        start = self.startq * 100
        ansl1 = []
        ansl2 = []

        # for each q
        for q in range(self.startq, self.q_num):
            # for q in [23, 29, 62, 81, 86, 87]:
            #     for index, i in enumerate(self.q_ids):
            #         if i == q:
            #             break

            temp_q_id = self.q_ids[q * 100]
            temp_q_tuple = self.q_dict[temp_q_id]
            temp_doc_ids = self.doc_ids[start:start + 100]

            # temp_q_tuple = self.q_dict[q]

            # temp_doc_ids = self.doc_ids[q * 100:q * 100 + 100]
            # temp_doc_ids = self.doc_ids[index:index + 100]
            start += 100
            print('index: ', q, 'temp_q_id:', temp_q_id)

            # 滑动窗口
            temp_d = defaultdict(bool)
            for endindex in range(self.round):  # topN
                start_w = self.samples - self.window_size + 1  # eq 只排前30，self.sample = 30， self.window = 2; 每趟都从后往前
                # start_w = 0
                while start_w >= endindex:
                    print('strat_w: ', start_w)
                    # content = self.getprompt_rankgpt(temp_q_tuple, temp_doc_ids[start_w:start_w + self.window_size])
                    # print(content)
                    # break
                    if temp_d[str(temp_doc_ids[start_w]) + '-' + str(temp_doc_ids[start_w + 1])]:  # 剪枝
                        start_w -= self.step_size
                        continue
                    content = self.getprompt_rankgpt(temp_q_tuple, temp_doc_ids[start_w:start_w + self.window_size])
                    print(content)
                    print('-------------chat---------------')
                    window_ans = self.chat(content)
                    print('answer ', window_ans)

                    # 异常结果处理
                    if window_ans[0] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                        p = window_ans.find('Passage')
                        if p != -1:
                            window_ans = window_ans[p + 7:]

                        I = window_ans.find('Identifier')
                        if I != -1:
                            window_ans = window_ans[I + 10:]

                        I = window_ans.find(':')
                        if I != -1:
                            window_ans = window_ans[I + 1:]

                        l = window_ans.find('[')
                        r = window_ans.find(']')
                        if l != -1:
                            window_ans = ''.join(window_ans[l + 1:r].split())
                    window_ans = int(window_ans)
                    # 标记 to 剪枝
                    if temp_doc_ids[start_w] == window_ans:
                        temp_d[str(temp_doc_ids[start_w]) + '-' + str(temp_doc_ids[start_w + 1])] = True
                    else:
                        second = temp_doc_ids[start_w]
                        temp_doc_ids[start_w] = window_ans
                        temp_doc_ids[start_w + 1] = second
                    start_w -= self.step_size

                # print('temp_doc_ids : ', temp_doc_ids)
            print('temp_doc_ids : ', temp_doc_ids)
            print('len(temp_doc_ids) : ', len(temp_doc_ids))

            ansl1.extend([temp_q_id] * len(temp_doc_ids))
            ansl2.extend(temp_doc_ids)
            pd.DataFrame({'qid': ansl1, 'docid': ansl2}).to_csv(
                './prompting_reranking_results/cricket_players_v4_180_random_pairwise_ans_' + str(self.startq) + '.csv', index=False)


if __name__ == '__main__':
    reranker = ReRanker('cricket_players_v4_180.csv', 20, 2, 0)
    reranker.pairwise_rank()

