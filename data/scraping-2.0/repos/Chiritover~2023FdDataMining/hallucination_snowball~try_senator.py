import json
import os
import openai
import time
import re

data_path = 'data/senator_search.json'
with open(data_path, 'r') as f:
    data = json.load(f)
print(len(data))

openai_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_key

for questions in data:
    try: 
        completion = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo-0613',
            messages = [{'role':'user','content':questions}],
        )
    except:
        time.sleep(60)
        completion = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo-0613',
            messages = [{'role':'user','content':questions}],
        )
    answer = completion.choices[0].message['content']
    # 将answer写入json文件
    with open('./result/senator_answer_original_gpt3.5_0613.json', 'a+') as f:
        query = {'question':questions, 'answer':answer}
        b = json.dumps(query)
        f.write(b)
        f.write('\n')