# -*- coding: utf-8 -*-
# @Author  : Zijian li
# @Time    : 2023/5/5 14:43
import pandas as pd
from snownlp import SnowNLP
from tqdm.auto import tqdm
import time
import openai
import os
import pinecone


def processNotionData():
    notion_data = pd.read_csv('../notion_data.csv')
    texts = notion_data['text_content'].tolist()
    titles = notion_data['title'].tolist()
    page_urls = notion_data['page_url'].tolist()
    page_ids = notion_data['page_id'].tolist()

    insert_data = []

    for i in tqdm(range(len(texts))):
        text = texts[i]
        title = titles[i]
        page_url = page_urls[i]
        page_id = page_ids[i]
        s = SnowNLP(text)
        sentences = s.sentences
        for sentence in sentences:
            insert_data.append({
                'text': sentence,
                'title': title,
                'page_url': page_url,
                'page_id': page_id
            })
    return insert_data


def init_pinecone():
    # initialize connection to pinecone (get API key at app.pinecone.io)

    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_ENV = os.environ.get('PINECONE_ENV')

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV  # 需要配置http proxy
    )

    index_name = "openai-notion"

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # If index does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine',
            metadata_config={'indexed': ['title', 'page_url', 'page_id', 'original_text']}
        )
    # connect to index
    index = pinecone.Index(index_name)
    # view index stats
    index.describe_index_stats()

    return index


def get_embedding(text, model="text-embedding-ada-002", delay=1, max_retries=3):
    # 防止请求过快，限制1min内只能发送60个请求
    time.sleep(1)
    retries = 0
    while retries < max_retries:
        try:
            embed_data = openai.Embedding.create(input=text, model=model)
            return embed_data['data']
        except openai.error.APIError as error:
            # 如果出现请求过于频繁的错误，等待指定时间再尝试发送请求
            if error.status == 429:
                time.sleep(delay)
                retries += 1
                continue
            else:
                raise error
    raise Exception("Maximum number of retries exceeded.")


def create_vectors(data_list):
    # 根据笔记内容按照相应格式创建向量
    no = 0
    vec_list = []
    for data in tqdm(data_list):
        text = data['text']
        embed = get_embedding(text)[0]['embedding']
        meta_data = {
            'title': data['title'],
            'page_url': data['page_url'],
            'page_id': data['page_id'],
            'original_text': text
        }
        vec = {
            'id': data['page_id'] + str(no),
            'values': embed,
            'metadata': meta_data,
        }
        vec_list.append(vec)
        no += 1
    return vec_list
