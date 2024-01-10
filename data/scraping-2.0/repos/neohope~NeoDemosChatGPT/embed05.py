#!/usr/bin/env python3
# -*- coding utf-8 -*-

'''
读取toutiao_cat_data.txt
进行预处理
抽取1000条
按每100条进行批处理，得到向量化结果
结果保存到toutiao_cat_data_all_with_embeddings.parquet

比上一个例子：
1、增加了批处理支持
2、增加了延时backoff
3、序列化为parquet
'''

import openai
import backoff
import tiktoken
import pandas as pd
import yaml
from openai.embeddings_utils import get_embeddings


# 向量化参数
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
batch_size = 1000


def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


# 调用频繁后，有可能被封，开启backoff功能
# 调用batch接口
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
    return embeddings


if __name__ == '__main__':
    get_api_key()

    # 数据读入
    df = pd.read_csv('data/toutiao_cat_data.txt', sep='_!_', names=['id', 'code', 'category', 'title', 'keywords'])
    # 填充null
    df = df.fillna("")
    df["combined"] = (
        "标题: " + df.title.str.strip() + "; 关键字: " + df.keywords.str.strip()
    )
    print("Lines of text before filtering: ", len(df))
    
    # 删除超长的例子
    encoding = tiktoken.get_encoding(embedding_encoding)
    df["n_tokens"] = df.combined.apply(lambda x: len(encoding.encode(x)))
    df = df[df.n_tokens <= max_tokens]
    print("Lines of text after filtering: ", len(df))
    
    # 随机抽取1000条
    df_1k = df.sample(1000, random_state=42)

    # 按100条分组进行批处理
    prompts = df_1k.combined.tolist()
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    # 进行批处理
    embeddings = []
    for batch in prompt_batches:
        batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
        embeddings += batch_embeddings

    # 向量化结果保存为parquet格式
    df_1k["embedding"] = embeddings
    df_1k.to_parquet("data/toutiao_cat_data_all_with_embeddings.parquet", index=True)

