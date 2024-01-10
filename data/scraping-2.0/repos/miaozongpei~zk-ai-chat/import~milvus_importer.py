from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Milvus
from langchain.text_splitter import CharacterTextSplitter

import os
import time  # 导入 time 模块

import config
from embedding.xinghuo_embedding import XhEmbeddings
import loader.base_loader
from langchain.embeddings import DashScopeEmbeddings


#embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
#embeddings =XhEmbeddings(appid=config.embedding_xh_appid, api_key=config.embedding_xh_api_key,api_secret=config.embedding_xh_api_secret,embedding_url=config.embedding_xh_embedding_url)
embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=config.llm_tyqw_api_key)
vector_db = Milvus(
    embedding_function=embeddings,
    connection_args={"host": config.Milvus_host, "port": config.Milvus_port, "user": config.Milvus_user,
                     "password": config.Milvus_password},
)
def import_file_url(file_url, collection_name):
    docs = loader.base_loader.load_file(file_url)
    if docs is not None:
        text_splitter = CharacterTextSplitter(chunk_size=config.text_splitter_chunk_size, chunk_overlap=config.text_splitter_chunk_overlap)
        docs = text_splitter.split_documents(docs)
        for i in range(0, len(docs), 10):
            batch_docs = docs[i:i + 10]
            vector_db.collection_name = collection_name
            time.sleep(20)  # 休眠20秒

def import_file_path(file_path, collection_name):
    texts = []
    for filename in os.listdir(file_path):
        file_url = f'{file_path}/{filename}'
        import_file_url(file_url, collection_name)

#import_file_path("/Users/miao/mydocs/个人/公司","my_doc")
#import_file_url("/Users/miao/mydocs/个人/公司/公司资料/创影数字人产品介绍2023V1（中科数智人）.pptx","my_doc1")
#import_file_url("/Users/miao/mydocs/个人/公司/6年级数学知识点/2.jpg","suxue6")
#import_file_path("/Users/miao/mydocs/个人/公司/育儿", "yuer3")

import_file_path("/Users/miao/mydocs/个人/公司/邮乐3","my_doc4")

#import_file_url("/Users/miao/mydocs/个人/公司/邮乐2/微调.doc","my_doc2")