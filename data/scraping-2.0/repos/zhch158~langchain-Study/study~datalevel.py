import os
from langchain.chat_models import ChatOpenAI as OpenAI

import openai
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index import LLMPredictor, SimpleDirectoryReader, VectorStoreIndex, LangchainEmbedding, ServiceContext, Document
import logging
import sys
# 从同级目录下的utils目录中导入setup_workdir和setup_env函数
# sys.path.insert(0, os.path.expanduser("~")+"/langchain-ChatGLM")
sys.path.insert(0, os.path.dirname(__file__) + "/..")
from utils.base import setup_env, setup_workdir

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=LOG_FORMAT, encoding='utf-8')
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# get_data_level函数说明如下：
# 1. 从data_level.txt中读取数据，每一行为一个文档，每个文档之间用\n\n分割
# 2. 通过LangchainEmbedding加载模型，这里使用的是sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# 3. 通过VectorStoreIndex.from_documents构建索引
# 4. 通过index.as_query_engine构建query_engine
# 5. 通过query_engine.query(query)进行查询
def get_data_level(query):
    # define LLM model
    llm_model = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo-0613", max_tokens=2048))
    
    # embed_model = LangchainEmbedding(
        # HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese"))
    service_context = ServiceContext.from_defaults(
        embed_model=embed_model,
        llm_predictor=llm_model,
    )

    # documents = SimpleDirectoryReader(input_files=['data_level.txt']).load_data()
    texts = open('data_level.txt', 'r', encoding='utf-8').read().split('\n\n')
    # 生成Document对象，过滤空文档
    documents = list()
    for text in texts:
        text=text.strip()
        if text != '':
            documents.append(Document(text))
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # index.storage_context.persist()
    query_engine = index.as_query_engine(similarity_top_k=3)

    result = query_engine.query(query)
    return result


if __name__ == '__main__':
    # os.environ['OPENAI_API_KEY'] = ""
    # openai.api_key =  os.environ['OPENAI_API_KEY']

    setup_workdir(os.path.dirname(__file__))
    setup_env()

    query = "请说明客户信息表中，身份证号，吸烟史，是否患有糖尿病等属性属于什么安全级别?"
    results = get_data_level(query)
    print("======================")
    print(results)
