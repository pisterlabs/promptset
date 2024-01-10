# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:18:06 2023

@author: 74719
"""
import openai
"""

openai.api_key = "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"
def generate_data_by_prompt(prompt):
  response = openai.Completion.create(
    engine = "text-davinci-003",
    prompt = prompt,
    temperature = 0.5,
    max_tokens = 2048,
    # 只返回一条结果 
    top_p = 1
  ) 
  return response.choices[0].text
prompt = """
#请你生成50条淘宝网里的商品的标题，每条在30个字左右，品类是3C数码产品，标题里往往
#也会有一些促销类的信息，每行一条。
"""
# 分步 
data = generate_data_by_prompt(prompt)
print(data)


import pandas as pd
#由txt文件变为一个数组
product_names = data.strip().split('\n')#空格转为换行符
print(product_names)
#二维excel表格  
df=pd.DataFrame({'product_name':product_names})
df.head()

df.product_name = df.product_name.apply(lambda x:x.split('.')[1].strip())
df.head()
"""
#keyword 手机 如何 拿出50 条手机的数据
#相似度 向量表达
from openai.embeddings_utils import get_embeddings,cosine_similarity

#print(get_embeddings('加菲猫', 'text-embedding-ada-002'))#处理模型


# 情感分类
from openai.embeddings_utils import get_embedding, cosine_similarity
EMBEDDING_MODEL = "text-embedding-ada-002"#嵌入式向量模型，专门进行向量表达

positive_review = get_embedding("好评",EMBEDDING_MODEL)
print(positive_review)
negative_review = get_embedding("差评",EMBEDDING_MODEL)
positive_example = get_embedding("进哥哥拍照水平一流，财大第一")
print(positive_example)
#计算两者的值
def get_score(sample_embedding):
  return cosine_similarity(sample_embedding, positive_review) - cosine_similarity(sample_embedding, negative_review)

positive_score = get_score(positive_example)

print("好评例子的评分: %f" % (positive_score))


