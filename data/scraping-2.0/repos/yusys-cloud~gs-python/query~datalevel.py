import os

import openai
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import SimpleDirectoryReader, VectorStoreIndex, LangchainEmbedding, ServiceContext, Document
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# get_data_level函数说明如下：
# 1. 从data_level.txt中读取数据，每一行为一个文档，每个文档之间用\n\n分割
# 2. 通过LangchainEmbedding加载模型，这里使用的是sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# 3. 通过VectorStoreIndex.from_documents构建索引
# 4. 通过index.as_query_engine构建query_engine
# 5. 通过query_engine.query(query)进行查询
def get_data_level(query):
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
    )

    # documents = SimpleDirectoryReader(input_files=['data_level.txt']).load_data()
    texts = open('data_level.txt', 'r', encoding='utf-8').read().split('\n\n')
    documents = [Document(text) for text in texts]
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist()
    query_engine = index.as_query_engine(similarity_top_k=5)

    result = query_engine.query(query)
    return result


if __name__ == '__main__':
    os.environ['OPENAI_API_KEY'] = "sk-xxxx"
    openai.api_key =  os.environ['OPENAI_API_KEY']

    query = "请说明客户信息表中，身份证号，吸烟史，是否患有糖尿病等属性属于什么安全级别?"
    results = get_data_level(query)
    print("======================")
    print(results)
