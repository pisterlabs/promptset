import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.environ.get("MONGODB"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
CHAT_MODEL = os.environ.get("CHAT_MODEL")
CHROMA_ROOT = os.environ.get("CHROMA_ROOT")
DOC_EMBED_COLLECTION = os.environ.get("DOC_EMBED_COLLECTION")


def search_doc(query: str):
    persist_directory = os.path.join(CHROMA_ROOT, DOC_EMBED_COLLECTION)

    embeddings = OpenAIEmbeddings()
    chroma = Chroma(
        collection_name=DOC_EMBED_COLLECTION,
        embedding_function=embeddings,
        # client_settings=client_settings,
        persist_directory=persist_directory,
    )
    n_docs = len(chroma._collection.get()["documents"])
    print(n_docs)
    docs = chroma.similarity_search_with_score(query=query, k=min(5, n_docs))
    result = [doc[0].metadata["_id"] for doc in docs]
    return result
