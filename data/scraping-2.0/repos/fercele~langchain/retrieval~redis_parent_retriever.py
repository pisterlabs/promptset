import os
from typing import List     

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.redis import Redis #VectorStore para armazenar os filhos
from langchain.retrievers import ParentDocumentRetriever

from .redis_doc_store import RedisJSONStore

CWD = os.getcwd()

#Obrigatório no construtor do parent retriever, mas não usado no app, pois o retriever será somente leitura
MAX_CHUNK_SIZE = 1210
CHUNK_OVERLAP = 256

REDIS_INDEX_NAME = os.getenv("REDIS_INDEX_NAME")
REDIS_SCHEMA_PATH = os.path.join(CWD, "db_redis", "redis_schema.yaml")

#Todos os parent chunks tem chaves no redis com esse prefixo
REDIS_PARENT_KEY_PREFIX = "parentdoc"


def get_redis_url():
    host = os.getenv("REDIS_HOST")
    port = os.getenv("REDIS_PORT")
    password = os.getenv("REDIS_PW")
    return f"redis://:{password}@{host}:{port}"


def get_redis_store(embeddings_model:OpenAIEmbeddings) -> Redis:
    redis_vector_store = Redis.from_existing_index(
        embedding=embeddings_model,
        redis_url=get_redis_url(),
        index_name=REDIS_INDEX_NAME,
        schema=REDIS_SCHEMA_PATH
    )
    
    return redis_vector_store


def get_redis_parent_retriever() -> ParentDocumentRetriever:
    #Este será usado para quebrar cada parent conforme o tamanho máximo recomendado de tamanho de chunk para geração de embedding
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size = MAX_CHUNK_SIZE,
        chunk_overlap  = CHUNK_OVERLAP,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""] #Prioriza ponto final antes de espaço.
    )

    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    redis_vector_store:Redis = get_redis_store(embeddings_model)

    #O namespace é o prefixo de todas as chaves de parent document
    redis_doc_store:RedisJSONStore = RedisJSONStore(redis_url=get_redis_url(), namespace=REDIS_PARENT_KEY_PREFIX)

    retriever = ParentDocumentRetriever(
        vectorstore=redis_vector_store,
        docstore=redis_doc_store,
        child_splitter=child_splitter,
        search_kwargs={"k": int(os.getenv("TOP_K"))}
    )

    return retriever

