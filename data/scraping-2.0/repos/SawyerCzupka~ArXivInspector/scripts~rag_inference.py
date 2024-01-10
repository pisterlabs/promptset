import logging
logging.basicConfig(level=logging.DEBUG)

from langchain.llms import TextGen
from llama_index import ServiceContext, set_global_service_context
from llama_index import VectorStoreIndex
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import LangChainLLM, OpenAI
from llama_index.retrievers import VectorIndexRetriever
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import QdrantVectorStore, VectorStoreQuery
from llama_index.query_engine import RetrieverQueryEngine
from qdrant_client import QdrantClient


# llm = LangChainLLM(llm=TextGen(model_url="http://localhost:5000"))
llm = OpenAI(api_base="http://127.0.0.1:5000/v1", api_key="sk-111111111111111111111111111111111111111111111111")
embed_model = HuggingFaceEmbedding(model_name="jinaai/jina-embeddings-v2-base-en", trust_remote_code=True) # Trust set to true for Jina


service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

set_global_service_context(service_context=service_context)


client = QdrantClient(url="http://localhost:6333")
qdrant = QdrantVectorStore(client=client, collection_name="arxiv-2023")
storage_context = StorageContext.from_defaults(vector_store=qdrant)
index = VectorStoreIndex.from_vector_store(vector_store=qdrant, storage_context=storage_context)

retriever = VectorIndexRetriever(index=index, similarity_top_k=5)


query_engine = RetrieverQueryEngine.from_args(retriever=retriever, service_context=service_context)


def rag_query(query: str):
    query_embedding = embed_model.encode(query)
    vector_store_query = VectorStoreQuery(query_embedding=query_embedding)
    query_result = qdrant.query(vector_store_query)
    return query_result


def query_with_engine(query_str: str):
    query_result = query_engine.query(query_str)
    return query_result


if __name__ == '__main__':
    query = "What is the latest research on LLMs?"
    result = query_with_engine(query)
    print(result)
