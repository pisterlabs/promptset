#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai, backoff, yaml
import faiss
import pandas as pd
import numpy as np
from IPython.display import display
from openai.embeddings_utils import get_embeddings, get_embedding, cosine_similarity

"""
生产训练数据
并使用embeding进行推荐
采用了faiss进行加速
数据量巨大的时候，可以使用向量数据库：pinecone、weaviate、milvus
"""

# 模型参数
embedding_model = "text-embedding-ada-002"
batch_size = 100


def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]

COMPLETION_MODEL = "text-davinci-003"


# 生成测试数据
def generate_data_by_prompt(prompt):
    response = openai.Completion.create(
        engine=COMPLETION_MODEL,
        prompt=prompt,
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
    )

    r = response.choices[0].text.strip().split('\n')
    return r


# 批处理获取向量
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
    return embeddings


# 试用faiss加速
def load_embeddings_to_faiss(df):
    embeddings = np.array(df['embedding'].tolist()).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# 商品查找最相似的前n个商品
def search_index(index, df, query, n=5):
    query_vector = np.array(get_embedding(query, engine=embedding_model)).reshape(1, -1).astype('float32')
    distances, indexes = index.search(query_vector, n)

    results = []
    for i in range(len(indexes)):
        product_names = df.iloc[indexes[i]]['product_name'].values.tolist()
        results.append((distances[i], product_names))    
    return results


# 根据现有商品推荐最相似的前n个商品
def recommend_index(index, df, product_name, n=3):
    query_vector = np.array(df[df['product_name'] == product_name].iloc[0].embedding).reshape(1, -1).astype('float32')
    distances, indexes = index.search(query_vector, n)

    results = []
    for i in range(len(indexes)):
        product_names = df.iloc[indexes[i]]['product_name'].values.tolist()
        results.append((distances[i], product_names))    
    return results


if __name__ == '__main__':
    get_api_key()

    roduct_prompt = """请你生成50条淘宝网里的商品的标题，每条在30个字左右，品类是3C数码产品，标题里往往也会有一些促销类的信息，每行一条。"""
    product_names = generate_data_by_prompt(roduct_prompt)
    df = pd.DataFrame({'product_name': product_names})
    df.product_name = df.product_name.apply(lambda x: x.split('.')[1].strip())
    df.head()

    clothes_prompt = """请你生成50条淘宝网里的商品的标题，每条在30个字左右，品类是女性的服饰箱包等等，标题里往往也会有一些促销类的信息，每行一条。"""
    clothes_product_names = generate_data_by_prompt(clothes_prompt)
    clothes_df = pd.DataFrame({'product_name': clothes_product_names})
    clothes_df.product_name = clothes_df.product_name.apply(lambda x: x.split('.')[1].strip())
    clothes_df.head()

    # 合并上述信息，生成实验数据
    df = pd.concat([df, clothes_df], axis=0)
    df = df.reset_index(drop=True)
    display(df)

    # 构造批处理promot
    prompts = df.product_name.tolist()
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    # 批处理获取向量
    embeddings = []
    for batch in prompt_batches:
        batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
        embeddings += batch_embeddings

    # 序列化为parquet
    df["embedding"] = embeddings
    df.to_parquet("data/taobao_product_title.parquet", index=False)

    # 试用faiss进行加速处理
    index = load_embeddings_to_faiss(df)
    
    # 商品查找
    products = search_index(index, df, "自然淡雅背包", k=3)
    for distances, product_names in products:
        for i in range(len(distances)):
            print(product_names[i], distances[i])

    # 商品推荐
    results = recommend_index(index, df, "【限量】OnePlus 7T Pro 8GB+256GB 全网通", n=3)

