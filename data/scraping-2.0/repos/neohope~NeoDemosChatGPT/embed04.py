#!/usr/bin/env python3
# -*- coding utf-8 -*-

import pandas as pd
import tiktoken
import openai
import yaml
from openai.embeddings_utils import get_embedding, get_embeddings


'''
读取toutiao_cat_data.txt
进行预处理
抽取1000条进行向量化
结果保存到toutiao_cat_data_all_with_embeddings.csv
'''


def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


# 向量化参数
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191


if __name__ == '__main__':
    get_api_key()

    # 读入数据
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

    # get embedding and save the result
    df_1k["embedding"] = df_1k.combined.apply(lambda x : get_embedding(x, engine=embedding_model))
    df_1k.to_csv("data/toutiao_cat_data_all_with_embeddings.csv", index=False)
