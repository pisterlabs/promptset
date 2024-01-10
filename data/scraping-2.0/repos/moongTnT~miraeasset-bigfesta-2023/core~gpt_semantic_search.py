import os, json

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

import pandas as pd

from data.fetch_data import fetch_pdf_info


def get_vectordb():
    persist_directory='db'

    embedding = OpenAIEmbeddings(
        model='text-embedding-ada-002'
    )

    vectordb = Chroma( # 기존 벡터 DB 로드
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    
    return vectordb


def get_filter(etf_tkr='AIQ'):

    pdf = fetch_pdf_info(etf_tkr=etf_tkr)

    pdf_df = pd.DataFrame(pdf)

    filter_list = []

    for i, row in pdf_df.iterrows():

        my_dict = {}

        symbol = row.child_stk_tkr

        my_dict['symbol'] = symbol

        filter_list.append(my_dict)
        
    return filter_list


def get_similar_symbols(*args, **kwargs):
    
    vectordb = kwargs.pop('vectordb')
    keyword = kwargs.pop('keyword')
    filter_list = kwargs.pop('filter_list')
    k = kwargs.pop('k', 5)

    docs = vectordb.similarity_search_with_score(keyword, k=k, filter={
        '$or': filter_list
        })
    
    symbol_list = []

    for doc in docs:
        symbol_list.append(doc[0].metadata['symbol'])
        
    return symbol_list