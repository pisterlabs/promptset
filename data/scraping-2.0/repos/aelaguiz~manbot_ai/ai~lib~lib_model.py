import logging
from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI, ChatOllama
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
import pinecone
from langchain.vectorstores import Pinecone

_vectordb = None
_embedding = None
_pinecone_index = None
_llm = None
_embedding = None


def init(model_name, api_key, temp=0.5):
    global _llm
    global _embedding

    logger = logging.getLogger(__name__)

    if _llm:
        logger.warning("LLM already initialized, skipping")
        return _llm

    _llm = ChatOpenAI(model_name=model_name, temperature=temp)
    _embedding = OpenAIEmbeddings(openai_api_key=api_key, timeout=30)

    return _llm


def get_embedding_fn():
    global _embedding

    logger = logging.getLogger(__name__)

    if not _embedding:
        logger.error("Embedding not initialized, call init() first")
        raise Exception("Embedding not initialized, call init() first")
    
    return _embedding

# def get_vectordb():
#     global _vectordb
#     global _pinecone_index

#     if _vectordb:
#         return _vectordb

#     logging.info(f"Using Pinecone")
#     pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
#     pinecone_index = pinecone.Index(index_name=os.getenv("PINECONE_INDEX_NAME"))

#     _vectordb = Pinecone(pinecone_index, get_embedding_fn(), "text")

#     return _vectordb

def get_llm():
    global _llm

    logger = logging.getLogger(__name__)

    if not _llm:
        logger.error("LLM not initialized, call init() first")
        raise Exception("LLM not initialized, call init() first")

    return _llm