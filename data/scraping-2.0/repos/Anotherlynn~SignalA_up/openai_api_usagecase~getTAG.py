# #!./.conda/envs/py310/bin/python3.10

# This is the file of using OpenAI APIs
# author: Xinyi Li
# contact: xl4412@nyu.edu
# time of completion: 2023-08-10

# 尝试接口openai的api
import pandas as pd
import tiktoken
import openai
import json
import os
import re
import datetime
import time


from langchain.chat_models import ChatOpenAI

from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index import set_global_service_context


def get_embedding(text, model='text-embedding-ada-002'):
    text = text.replace("\n"," ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
#
#
# dfa = pd.read_csv('data/report_clean.csv')
# dfa = dfa[['TradingDate','InfoTitle','Detail','BulletinType','InnerCode','SecuCode']].reset_index(drop=True)
# dfa = dfa.sort_values("TradingDate").reset_index(drop=True)
#
# group = dfa.groupby('InnerCode')
# top_n = 10
# # 保留最近的10条做测试
# df_test = group.tail(top_n)
# # 使用训练集的最后50条分类
# df = dfa.drop(df_test.index).groupby('InnerCode').tail(top_n*5)
# df = df.dropna()
# df["combined"] = (
#     df.InfoTitle.str.strip() + " " + df.Detail.str.strip()
# )
# # df.drop("Time", axis=1, inplace=True)
# embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
# max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
#
# encoding = tiktoken.get_encoding(embedding_encoding)
#
# # omit reviews that are too long to embed
# df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
# df = df[df.n_tokens <= max_tokens]
# len(df)
# # df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model)



prompt = "在分红公告，股东大会决议公告，业绩预告公告，持股变动公告，资产重组公告，再融资公告，股权激励公告，关联交易公告，担保公告，退市风险公告，交易所交易公开信息公告，现金管理公告，会计政策变更公告，人员聘请公告，审计保留意见公告，IPO公告，变更信息公告，内部控制公告，新项目开展公告这19个标签中，给出下面公告的最可能标签："
completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": prompt.strip()},
  ]
)


llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",openai_api_key=api_key)
service_context = ServiceContext.from_defaults(llm=llm)
set_global_service_context(service_context=service_context)
doc = SimpleDirectoryReader(input_files=["上市公司.pdf"]).load_data()
index = VectorStoreIndex.from_documents(doc,service_context=service_context)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from llama_index.prompts import Prompt

chat_text_qa_msgs = [
    HumanMessagePromptTemplate.from_template(
        "填写公告的分类和组成的表格.\n| 公告类型 | 公告内容 |"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
    ),
]
chat_text_qa_msgs_lc = ChatPromptTemplate.from_messages(chat_text_qa_msgs)
text_qa_template = Prompt.from_langchain_prompt(chat_text_qa_msgs_lc)

# Refine Prompt
chat_refine_msgs = [
    SystemMessagePromptTemplate.from_template(
        "Always answer the question, even if the context isn't helpful."
    ),
    HumanMessagePromptTemplate.from_template(
        "We have the opportunity to refine the original answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context, refine the original answer to better "
        "answer the question: {query_str}. "
        "If the context isn't useful, output the original answer again.\n"
        "Original Answer: {existing_answer}"
    ),
]
from llama_index.schema import Node
from llama_index.response_synthesizers import get_response_synthesizer

chat_refine_msgs_lc = ChatPromptTemplate.from_messages(chat_refine_msgs)
refine_template = Prompt.from_langchain_prompt(chat_refine_msgs_lc)

def time_printer():
    now = datetime.datetime.now()
    ts = now.strftime('%Y-%m-%d %H:%M:%S')
    print('do func time :', ts)


def loop_monitor(dff):
    i = 0
    while True and i < 2400:
        time_printer()
        print(i)
        if i + 253 <= 2271:
            xx = dff.iloc[i:i + 253]
        else:
            xx = dff.iloc[i:]
        xx['Tag'] = xx.apply(lambda x: classify_tag(x), axis=1)
        print("sucess")
        listA.append(xx)
        i += 200
        time.sleep(650)


def load_api_key(secrets_file):
    with open(secrets_file) as f:
        secrets = json.load(f)
    return secrets["OPENAI_API_KEY"]


def request_completion(prompt):
    completion_response = openai.ChatCompletion.create(
        messages=prompt,
        temperature=0,
        max_tokens=10,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )

    return completion_response


def classify_tag(df):
    detail = re.sub('([^\u4e00-\u9fa5\u0030-\u0039])', '', df['Detail'])
    if len(detail) > 1500:
        prompt = detail[:1500]
    else:
        prompt = detail
    zero_shot_prompt = [
        {
            "role": "system",
            "content": "你会被提供一则公司公告。你的任务是给出公告的唯一的类别。可用的选项有：分红，股东大会决议，业绩预告，持股变动，资产重组，再融资，股权激励，关联交易，担保，退市风险，交易所交易公开信息，现金管理，会计政策变更，人员聘请，审计保留意见，IPO，变更信息，内部控制，新项目开展。如果无法分类，说无法分类。只回答类别名。"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    classification = request_completion(zero_shot_prompt)['choices'][0]['message']['content']

    return classification


## Zero-shot Classification


api_key = load_api_key("../proj/secretS.json")
openai.api_key = api_key

COMPLETIONS_MODEL = "gpt-3.5-turbo-0613"

df = pd.read_csv("../data/sample.csv")
listA = []
loop_monitor(df)
df_with_tag = pd.concat(listA)
df_with_tag.to_csv("data/sample_with_tag.csv", encoding='utf-8-sig', index=None)
