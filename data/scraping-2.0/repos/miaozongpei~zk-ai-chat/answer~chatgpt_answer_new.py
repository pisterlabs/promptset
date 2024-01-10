from langchain.vectorstores import Milvus
from langchain.chains.retrieval_qa.base import RetrievalQA
from typing import Any
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, FAISS
from langchain.schema import Document
import langchain


from embedding.xinghuo_embedding import XhEmbeddings
from llm.spark_llm import Spark
from llm.dashscope_llm import Dashscope

import config
from langchain.cache import RedisCache
from redis import Redis
from langchain.embeddings import DashScopeEmbeddings
#embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=config.llm_tyqw_api_key)
#embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
embeddings =XhEmbeddings(appid=config.embedding_xh_appid,api_key=config.embedding_xh_api_key,api_secret=config.embedding_xh_api_secret,embedding_url=config.embedding_xh_embedding_url)

# redis 缓存
langchain.llm_cache = RedisCache(redis_=Redis(host="8.130.178.88", port=6379, db=8))

#llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo-16k")
#llm = Dashscope()
llm = Spark(version=3)
def get_vector_chain(collection_name) -> Any:
    llm
    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:{question}'
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    Answer in Chinese，The answer cannot exceed 100.
    """

    #Answer in the language in which the question was asked:

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    vector_db = Milvus(
        embedding_function=embeddings,
        connection_args={"host": config.Milvus_host, "port": config.Milvus_port},
        collection_name=collection_name,
    )
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        },
    )
    return chain

def answer_bydoc(collection_name, question):
    chain = get_vector_chain(collection_name)
    return chain.run(question)
def answer_bybase(question):
    result = llm(question)
    return result

def query_doc(collection_name, question):
    vector_db = Milvus(
        embedding_function=embeddings,
        connection_args={"host": config.Milvus_host, "port": config.Milvus_port},
        collection_name=collection_name,
    )
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)
    return docs

def add_doc(collection_name, question,content):
    vector_db = Milvus(
        embedding_function=embeddings,
        connection_args={"host": config.Milvus_host, "port": config.Milvus_port},
        collection_name=collection_name,
    )
    doc = Document(page_content=content,
                   metadata={"source": question})
    docs=[]
    docs.append(doc)
    vector_db.add_documents(docs)


#eplay=answer("my_doc1","你们周六上班吗" )
#replay=answer("my_doc1","我周六可以去吗" )
#print(replay)

#replay=answer("my_doc1","你好" )
#print(replay)