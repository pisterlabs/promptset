from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from log_handler.Setup import logger

import json
import os

class ExactRetrieval():
    def __init__(self):
        self.api_key=None
        self.path_pdf=None
        self.chunk_size=None
        self.chunk_overlap=None
        self.embedding_model=None
        self.top_k=None
        self.read_config(config_path="./configs/configs.json")
    
    def read_config(self, config_path):
        try:
            config_file=open(config_path, "r")
        except IOError:
            logger.error('Specify the correct config path.')
        else:
            my_config=json.load(config_file)
            config_file.close()
            self.path_pdf=my_config['path_pdf']
            self.chunk_size_exact=my_config['chunk_size_exact']
            self.chunk_overlap=my_config['chunk_overlap']
            self.top_k=my_config['top_k']
    
    def load_and_split_contents(self):
        loader=PyPDFLoader(self.path_pdf)
        docs=loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=self.chunk_size_exact,
                                                     chunk_overlap=self.chunk_overlap,
                                                     add_start_index=True)
        result=text_splitter.split_documents(docs)
        final_result=[result[i].page_content.lower() for i in range(len(result))]
        return final_result
    
    def retrieve_doc(self, query):
        final_result=self.load_and_split_contents()
        scores=[]
        for i in final_result:
            if query in i:
                scores.append(1)
            else: scores.append(0)
        return [final_result[i] for i in range(len(final_result)) if scores[i]==1]
    
    
class openAI():
    def __init__(self, api_key):
        self.api_key=api_key
        self.path_pdf=None
        self.chunk_size=None
        self.chunk_size_exact=None
        self.chunk_overlap=None
        self.embedding_model=None
        self.top_k=None
        self.read_config(config_path="./configs/configs.json")
    
    def read_config(self, config_path):
        try:
            config_file=open(config_path, "r")
        except IOError:
            logger.error('Specify the correct config path.')
        else:
            my_config=json.load(config_file)
            config_file.close()
            self.path_pdf=my_config['path_pdf']
            self.chunk_size=my_config['chunk_size']
            self.chunk_size_exact=my_config['chunk_size_exact']
            self.chunk_overlap=my_config['chunk_overlap']
            self.embedding_model=my_config['embedding_model']
            self.top_k=my_config['top_k']

    def load_and_split_contents(self):
        loader=PyPDFLoader(self.path_pdf)
        docs=loader.load()
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                     chunk_overlap=self.chunk_overlap,
                                                     add_start_index=True)
        return text_splitter.split_documents(docs)
    
    def index_embedding(self, query:str):
        os.environ['OPENAI_API_KEY']=self.api_key
        loaded_docs=self.load_and_split_contents()
        embeddings=OpenAIEmbeddings(model=self.embedding_model)
        faiss_index=FAISS.from_documents(loaded_docs, embeddings)
        docs=faiss_index.similarity_search(query, k=self.top_k)
        docs_ret=[docs[i].page_content for i in range(len(docs))]
        return docs_ret
    
    
def query_builder_andRetrieve(query:str, api_key):
    tokenize_q=query.split()
    if len(tokenize_q)==1:
        retriver=ExactRetrieval()
        result=retriver.retrieve_doc(query.lower())
    else:
        retriver=openAI(api_key)
        result=retriver.index_embedding(query)
    return result
