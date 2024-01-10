# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:01:25 2023

@author: shangfr
"""

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import QianfanEmbeddingsEndpoint
from langchain.vectorstores import Chroma

def add_emb(docs,**kwargs):
    embeddings = QianfanEmbeddingsEndpoint()
    vectordb = Chroma.from_documents(
        collection_name=kwargs['collection_name'],
        documents=docs,
        embedding=embeddings,
        persist_directory=kwargs['directory'])  
    vectordb.persist()
    print("Vector DB init success! ")
    
def init_vectordb(file_path='data_csv/books_cn.csv', collection_name="fables_collection", directory='fables_db'):

    
    loader = CSVLoader(file_path, encoding='utf-8')
    docs = loader.load()

    max_tokens = 384
    docs_new = []
    for d in docs:
        if len(d.page_content)<max_tokens/1.3:
            docs_new.append(d)
        if len(docs_new)>15:
            add_emb(docs_new, collection_name="fables_collection", directory='fables_db')
            docs_new = []
    if docs_new:
        add_emb(docs_new, collection_name="fables_collection", directory='fables_db')


def load_vectordb(directory='fables_db', collection_name="fables_collection"):

    embeddings = QianfanEmbeddingsEndpoint()
    vectordb = Chroma(
        collection_name,
        embeddings,
        directory)
    
    return vectordb


#init_vectordb()
#vectordb = load_vectordb(directory='fables_db')
#retriever = vectordb.as_retriever(search_type="mmr")
#results =retriever.get_relevant_documents("猫和老鼠")[0]
#print(results.page_content)