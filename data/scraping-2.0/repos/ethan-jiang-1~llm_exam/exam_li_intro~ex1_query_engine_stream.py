from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage

import logging
import sys
import openai 
import os

def enable_debug():
    openai.log = "debug"

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def get_index():
    #documents = SimpleDirectoryReader('data').load_data()
    pss_data = "/root/llm_exam/llama_index/examples/paul_graham_essay/pss"
    if not os.path.isdir(pss_data):
        raw_data = "/root/llm_exam/llama_index/examples/paul_graham_essay/data"
        documents = SimpleDirectoryReader(raw_data).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=pss_data)

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=pss_data)
    # load index
    index = load_index_from_storage(storage_context)
    return index 


if __name__ == "__main__":
    #enable_debug()
    index = get_index()

    query_engine = index.as_query_engine(response_mode='tree_summarize', streaming=True)
    response = query_engine.query("What did the author do growing up?")
    print()
    response.print_response_stream()
    print()

    query_engine = index.as_query_engine(streaming=True)
    response = query_engine.query("What did the author do growing up?")
    print()
    response.print_response_stream()
    print()

    # chat_engine = index.as_chat_engine()
    # response = chat_engine.query("What did the author do growing up?")
    # print(response)