from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage

import logging
import sys
import openai 
import os

import pinecone
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores import PineconeVectorStore

def enable_debug():
    openai.log = "debug"

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def get_documents():
    raw_data = "/root/llm_exam/llama_index/examples/paul_graham_essay/data"
    documents = SimpleDirectoryReader(raw_data).load_data()
    return documents

# def get_index():
#     #documents = SimpleDirectoryReader('data').load_data()
#     pss_data = "/root/llm_exam/llama_index/examples/paul_graham_essay/pss"
#     if not os.path.isdir(pss_data):
#         documents = get_documents()
#         index = VectorStoreIndex.from_documents(documents)
#         index.storage_context.persist(persist_dir=pss_data)

#     # rebuild storage context
#     storage_context = StorageContext.from_defaults(persist_dir=pss_data)
#     # load index
#     index = load_index_from_storage(storage_context)
#     return index 

def init_pincone():
   ret = pinecone.init(api_key="3ffbc753-2b2b-47a8-9350-a470b1fe5691", environment="asia-southeast1-gcp-free")
   return ret


def create_index_in_pincone():
    # init pinecone
    pinecone.create_index("quickstart", dimension=1536, metric="euclidean", pod_type="p1")

    # construct vector store and customize storage context
    storage_context = StorageContext.from_defaults(
        vector_store=PineconeVectorStore(pinecone.Index("quickstart"))
    )

    # Load documents and build index
    documents = get_documents()
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index 

def connect_index_from_pincone():
    indexes = pinecone.list_indexes()
    if "quickstart" in indexes:
        try:
            vector_store = PineconeVectorStore(pinecone.Index("quickstart"))
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            return index
        except Exception as ex:
            print("exception occured", ex)
    return None


def get_pincone_index():
    #enable_debug()
    init_pincone()
    index = connect_index_from_pincone()
    if index is None:
        index = create_index_in_pincone()
    print(index)
    return index


def get_default_query_engine(index):
    return index.as_query_engine()


def get_cust1_query_engine(index):
    from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        vector_store_query_mode="default",
        filters=MetadataFilters(
            filters=[
                ExactMatchFilter(key="name", value="paul graham"),
            ]
        ),
        alpha=None,
        doc_ids=None,
    )
    return query_engine

def get_cust2_query_engine(index):
    from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
    from llama_index import get_response_synthesizer
    from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
    from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine

    # build retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,
        vector_store_query_mode="default",
        filters=[ExactMatchFilter(key="name", value="paul graham")],
        alpha=None,
        doc_ids=None,
    )

    # build query engine
    query_engine = RetrieverQueryEngine(
        retriever=retriever, response_synthesizer=get_response_synthesizer()
    )
    return query_engine


if __name__ == "__main__":
    index = get_pincone_index()

    default_query_engine = get_default_query_engine(index)
    response = default_query_engine.query("What did the author do growing up?")
    print()
    print(response)
    print()

    cust1_query_engine = get_cust1_query_engine(index)
    response = cust1_query_engine.query("What did the author do growing up?")
    print()
    print(response)
    print()

    # cust2_query_engine = get_cust2_query_engine(index)
    # response = cust2_query_engine.query("What did the author do growing up?")
    # print()
    # print(response)
    # print()