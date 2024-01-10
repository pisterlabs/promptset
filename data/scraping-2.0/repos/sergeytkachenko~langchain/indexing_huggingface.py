from langchain.retrievers import WeaviateHybridSearchRetriever

from docs_utils import get_academy_docs
from retrievers import get_weaviate_huggingface_retriever, weaviate_search


def indexing_huggingface() -> WeaviateHybridSearchRetriever:
    docs = get_academy_docs()
    retriever = get_weaviate_huggingface_retriever(docs)
    # retriever.add_documents(docs)
    return retriever

def search_huggingface(retriever: WeaviateHybridSearchRetriever, q: str):
    return weaviate_search(retriever, q)