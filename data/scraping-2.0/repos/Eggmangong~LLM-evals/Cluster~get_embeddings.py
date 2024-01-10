# 文件名：get_embeddings.py

import os
import openai
import pandas as pd
import numpy as np

# 设置OpenAI API密钥
openai.api_key = "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"

# 读取Excel文件
df = pd.read_excel('/Users/jinqigong/Desktop/Research/OpenAI Evals/new_prompt/original_prompt_complete.xlsx')
data = df[['Dataset', 'original_prompt']].values  # 假设你的数据集列名为'Dataset'，prompt列名为'original_prompt'

# 获取文本嵌入
embeddings = []
for Dataset, original_prompt in data:
    input_text = str(Dataset) + ' ' + str(original_prompt)  # 将数据集名称加入到prompt的开头
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=input_text
    )
    embeddings.append(response['data'][0]['embedding'])

# 将嵌入列表转换为numpy数组，并保存到文件中
embeddings = np.array(embeddings)
np.save('embeddings.npy', embeddings)
