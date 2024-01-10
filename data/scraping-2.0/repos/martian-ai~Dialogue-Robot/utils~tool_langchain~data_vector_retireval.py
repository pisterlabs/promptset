'''
#!/usr/bin/python3
# -*- encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2023 by Martain.AI, All Rights Reserved.
#
Description: # 
Author: # apollo2mars apollo2mars@gmail.com
################################################################################
'''


import os

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import chroma, Chroma

from constants import embedding_model_dict


def load_embedding_model(model_name="ernie-tiny"):
    """
    加载embedding模型
    :param model_name:
    :return:
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    """
    讲文档向量化，存入向量数据库
    :param docs:
    :param embeddings:
    :param persist_directory:
    :return:
    """
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    return db


def get_db(documents, embeddings=None):
    # 加载embedding模型
    embeddings = load_embedding_model('text2vec3')

    # 加载数据库
    if not os.path.exists('VectorStore'):
        print(">>>")
        documents = documents
        db = store_chroma(documents, embeddings)
    else:
        db = Chroma(persist_directory='VectorStore', embedding_function=embeddings)
    
    return db
