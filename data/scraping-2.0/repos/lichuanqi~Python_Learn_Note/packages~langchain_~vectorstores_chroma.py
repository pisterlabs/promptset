import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass, field

import pandas as pd
from langchain.document_loaders import TextLoader
from langchain.text_splitter import TokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


embedding_model_name = 'packages/langchain_/models/shibing624_text2vec-base-chinese'
persist_directory = 'packages/langchain_/vectordb'

embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
vectordb = Chroma(embedding_function=embeddings,
persist_directory=persist_directory)

# 增加一个文档
file_path = 'packages/langchain_/test.txt'
loader = TextLoader(file_path=file_path, encoding='utf-8')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
doc_texts = loader.load_and_split(text_splitter=text_splitter)
docids = vectordb.add_documents(documents=doc_texts)
vectordb.persist()

# 问题检索
query = "用2-3句话解释一下寄递市场库"
docs = vectordb.similarity_search_with_score(query)
print('检索问题: %s'%query)
pprint('检索结果: \n%s'%docs)