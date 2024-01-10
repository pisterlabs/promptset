# -*- coding: utf-8 -*-
"""
https://python.langchain.com/docs/modules/data_connection/indexing
https://www.langchain.com.cn/modules/indexes/getting_started

LangChain 主要关注于构建索引，目标是使用它们作为检索器。
"""
import os
import torch as th
from langchain.chains import RetrievalQA
from langchain.llms import (OpenAI, HuggingFacePipeline)
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator


print(th.cuda.get_device_name())  # NVIDIA GeForce GTX 1080 Ti
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# ----------------------------------------------------------------------------------------------------------------
# 路径
path_project = "C:/my_project/MyGit/Machine-Learning-Column/LangChain"
path_data = os.path.join(os.path.dirname(path_project), "data")
path_model = os.path.join(os.path.dirname(path_project), "model")

# ----------------------------------------------------------------------------------------------------------------
# 创建一行索引
loader = TextLoader('../state_of_the_union.txt', encoding='utf8')
index = VectorstoreIndexCreator().from_loaders([loader])

query = "What did the president say about Ketanji Brown Jackson"
index.query(query)

query = "What did the president say about Ketanji Brown Jackson"
index.query_with_sources(query)

