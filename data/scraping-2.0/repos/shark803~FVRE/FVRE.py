#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/12/24 09:41
# @Author  : Lei Wang
# @File    : FVRE.py
# @Project : FVRE
# @Software: PyCharm


import os
import openai
import json
import time

from openai import OpenAI
client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))

prompt_file = "./datasets/semeval/test/prompt.txt"

zeroshot_answer_file = "./datasets/semeval/test/zs_answer.txt"


#with open("./processed_dataset/processed_test.json","r") as f:
    #test_dataset = json.load(f)



# completion = client.chat.completions.create(
#     model="gpt-4-1106-preview",
#     messages=[
#         {"role": "system", "content": "You are an information extractor."},
#         {"role": "user", "content":"{}".format(prompt)},
#     ]
# )

with open(prompt_file,"r") as f, open(zeroshot_answer_file,"w") as f2:
    prompts = f.readlines()
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()
        completion = client.chat.completions.create(
            #model="gpt-4-1106-preview",
            model = "gpt-3.5-turbo-1106",

            messages=[
                {"role": "system", "content": "You are an information extractor."},
                {"role": "user", "content":"{}".format(prompt)},
            ]
        )
        time.sleep(1)
        print(completion.choices[0].message.content)
        f2.write(str(8001+i) + "\t"+ completion.choices[0].message.content+"\n")

