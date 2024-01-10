import joblib
from config import *
import openai
openai.api_version = "2023-03-15-preview"
from script.utils import calc_cos_similarity

user_query = "How to initialize Foxit PDF SDK"

user_query_response = openai.Embedding.create(
    input=user_query,
    engine="text-embedding-ada-002"
)
user_query_embedding = user_query_response['data'][0]['embedding']

text_embeddings = joblib.load('text_embeddings.pkl')

# 计算用户问题和每个段落的相似度，取相似度最高的几个段落，和用户的问题一起送入chatgpt
related_paragraph = calc_cos_similarity(text_embeddings, user_query_embedding)
print(f"最相关的章节为:{related_paragraph}")


conversation=[{"role": "system", "content": "You are an AI assistant that helps people find information."}]



response = openai.ChatCompletion.create(
    engine="ChatGPT-0301",
    messages=[
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information."
        },
        {
            "role": "user",
            "content": "根据上下文回答问题:\n对所有的pdf文档操作,必须要打开文档,打开文档的函数是openpdf,\n 最终关闭文档, 关闭文档的函数是closepdf\n 获取pdf页面总数的函数是getpageCount.\n 添加annot的函数是addpdfannot\n  问题:\n"
        }
    ],
    temperature=0.7,
    max_tokens=800,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None)
print(response['choices'][0]['message']['content'])
print(response)
