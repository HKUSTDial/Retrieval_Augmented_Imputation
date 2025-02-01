import time
import pandas as pd

from collections import defaultdict
import openai


openai.api_key = "xxxxxxxxx"
openai.api_base = "xxxxx"


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
        self.step_size = 20
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
        prompt.append('I will provide you with ' + str(
            self.window_size) + ' passages, each indicated by number identifier [identifier]. Rank them based on their relevance to query:{' + q_tuple + '} to infer the missing value N/A.')

        for i in range(self.window_size):
            # print(doc_ids[i])
            prompt.append('Passage [' + str(doc_ids[i]) + '] = {' + str(self.doc_dict[doc_ids[i]]) + '}')
        prompt.append(
            'The passages should be listed in descending order using identifiers, and the most relevant passages should be listed first. Only response the ranking results in the format of [identifier, identifier, ,..] , do not say any word or explain.')
        return '\n'.join(prompt)

    def listwise_rank(self):
        start = self.startq * 100
        ansl1 = []
        ansl2 = []
        # for each q
        for q in range(self.startq, self.q_num):

            temp_q_id = self.q_ids[q * 100]
            temp_q_tuple = self.q_dict[temp_q_id]
            temp_doc_ids = self.doc_ids[start: start + self.samples]
            # temp_doc_tuples = self.doc_tuples[start:start + self.samples]
            print('index: ', q, 'temp_q_id:', temp_q_id)
            start += 100

            # 滑动窗口
            start_w = self.samples - self.window_size - 50
            # start_w = 0
            for step in range((self.samples - 50 - self.window_size) // self.step_size + 1):
                print('strat_w: ', start_w)
                content = self.getprompt_rankgpt(temp_q_tuple, temp_doc_ids[start_w:start_w + self.window_size])
                print(content)
                print('-------------------chat-------------------')
                window_ans = self.chat(content)
                print(window_ans)
                l = window_ans.find('[')
                r = window_ans.find(']')
                window_ans = ''.join(window_ans[l + 1:r].split()).split(',')
                window_ans = list(map(int, window_ans))
                for j in range(len(window_ans)):
                    if not self.doc_dict[window_ans[j]]:
                        window_ans[j] = window_ans[j - 1]

                if len(window_ans) > self.window_size:
                    window_ans = window_ans[:self.window_size]
                if len(window_ans) < self.window_size:
                    bias_set = set(temp_doc_ids[start_w:start_w + self.window_size]) - set(window_ans)
                    for i in bias_set:
                        window_ans.append(i)
                        if len(window_ans) == self.window_size:
                            break
                print(temp_doc_ids[start_w:start_w + self.window_size])
                print(window_ans)

                temp_doc_ids[start_w:start_w + self.window_size] = window_ans[:]
                # print(temp_doc_ids[start_w:start_w + self.window_size])
                start_w -= self.step_size

                # break

            print(len(temp_doc_ids))

            ansl1.extend([temp_q_id] * len(temp_doc_ids))
            ansl2.extend(temp_doc_ids)
            pd.DataFrame({'qid': ansl1, 'docid': ansl2}).to_csv(
                './prompting_reranking_results/cricket_players_v4_180_random_listwise_ans_' + str(self.startq) + '.csv', index=False)


if __name__ == '__main__':
    reranker = ReRanker('cricket_players.csv', 100, 30, 120)
    reranker.listwise_rank()

