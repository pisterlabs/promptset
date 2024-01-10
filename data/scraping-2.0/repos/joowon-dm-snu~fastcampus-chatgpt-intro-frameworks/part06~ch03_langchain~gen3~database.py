import os
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "database", "chroma-persist")
CHROMA_COLLECTION_NAME = "fastcampus-bot"


_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)
_retriever = _db.as_retriever()


def query_db(query: str, use_retriever: bool = False) -> List[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs
