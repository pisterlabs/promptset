# https://learn.microsoft.com/en-us/azure/cognitive-services/openai/tutorials/embeddings
import openai
import os
import re
from num2words import num2words
import os
import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken

# s is input text
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()  
    return s

# search through the reviews for a specific product
def search_docs(df, engine_name, user_query, top_n=3, to_print=True):
    # engine should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    embedding = get_embedding(
        user_query,
        engine=engine_name 
    )
    df["similarities"] = df.ada_v2.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(top_n)
    )
    if to_print:
        print(res)
    return res

OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY") 
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") 

openai.api_type = "azure"
openai.api_key = OPENAI_KEY
openai.api_base = OPENAI_ENDPOINT
# openai.api_version = "2022-12-01"
openai.api_version = "2023-03-15-preview"

# url = OPENAI_ENDPOINT + "/openai/deployments?api-version=" + openai.api_version
# request_headers = {"api-key": OPENAI_KEY}
# # 发送请求，并输出详细请求信息和返回信息
# r = requests.get(url, headers=request_headers)
# # 解析返回的json，提取 data中的 model和id，再以表格形式打印出来
# # print(r.text)
# data = r.json()
# df = pd.DataFrame(data['data'])
# # 只输出 id 和 model 2 列
# df_id_model = df[['id','model']]
# print(df_id_model)

# This assumes that you have placed the bill_sum_data.csv in the same directory you are running Jupyter Notebooks
# print(os.getcwdb())
df=pd.read_csv('./AzureOpenAI/bill_sum_data.csv')
df_bills = df[['text','summary', 'title']]
# print(df_bills)
df_bills['text']= df_bills["text"].apply(lambda x : normalize_text(x))
tokenizer = tiktoken.get_encoding("cl100k_base")
df_bills['n_tokens'] = df_bills["text"].apply(lambda x: len(tokenizer.encode(x)))
df_bills = df_bills[df_bills.n_tokens<8192]
# print(df_bills)

sample_encode = tokenizer.encode(df_bills.text[0]) 
decode = tokenizer.decode_tokens_bytes(sample_encode)
print(len(decode))

# 注意这里要使用Deployment的名称，而不是Deployment使用的Model name: text-embedding-ada-002 (Version 2)
engine_embedding_name = "embedding"
df_bills['ada_v2'] = df_bills["text"].apply(lambda x : get_embedding(x, engine = engine_embedding_name)) 
# print(df_bills)

res = search_docs(df_bills, engine_embedding_name, "Can I get information on cable company tax revenue?", top_n=4)
# 显示第一条记录的summary
print(res.summary.iloc[0])