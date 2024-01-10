import json
import os
import openai
import time
import re

data_path = 'data/primality_testing.json'
with open(data_path, 'r') as f:
    data = json.load(f)
print(len(data))

openai_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_key

for questions in data:
    try: 
        completion = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo-0613',
            messages = [{'role':'user','content':questions['question']}],
        )
    except:
        time.sleep(60)
        completion = openai.ChatCompletion.create(
            model = 'gpt-3.5-turbo-0613',
            messages = [{'role':'user','content':questions['question']}],
        )
    answer = completion.choices[0].message['content']
    # 将answer写入json文件
    with open('./result/answer_original_gpt3.5_0613.json', 'a+') as f:
        query = {'question':questions['question'], 'answer':answer}
        b = json.dumps(query)
        f.write(b)
        f.write('\n')
        

   

