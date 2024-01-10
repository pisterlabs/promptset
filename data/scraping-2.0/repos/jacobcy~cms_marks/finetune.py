# -*- coding: utf-8 -*-

import openai
from setting import Excel

# 使用 API 密钥进行身份验证
openai.api_key = Excel.key

model = "curie:ft-silly-fish-2023-03-15-02-50-16"

question = r'程序员也危险了!GPT-4十秒即可生成一个网站'
prompt = [{'role': 'user', 'content': question}]

# 加载预训练模型
completion = openai.Completion.create(
    model=model,
    prompt=prompt)

print(completion)
