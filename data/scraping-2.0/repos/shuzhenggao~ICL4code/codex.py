import openai
import jsonlines
import json
import sys
from tqdm import tqdm
from time import sleep
import os


api_keys = ["sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]
fileanme = sys.argv[1]
api_idx = 0
openai.api_key = api_keys[api_idx]

querys = []
with jsonlines.open(fileanme) as reader:
    for obj in reader:
        querys.append(obj)

results = []
fail = []
for pos in tqdm(range(len(querys))):
    query = querys[pos]
    success = 0
    fail_count = 0
    while success!=1:
        try:
            response = openai.Completion.create(model="code-davinci-002",prompt=query['prompt'],temperature=0,max_tokens=256,top_p=1,frequency_penalty=0.0,presence_penalty=0.0,stop=["\n\n"])
            success=1
            result = {}
            result['label'] = query['label']
            result['choices'] = response["choices"]
            result['idx'] = pos
            with jsonlines.open(fileanme.split('.jsonl')[0]+'_results_.jsonl', mode='a') as f:
                f.write_all([result])
        except Exception  as e:
            info = e.args[0]
            print("Error: ", info)
            sleep(2)
            fail_count+=1
        if fail_count>50:
            fail.append(pos)
            break
    sleep(5)
print(api_idx)