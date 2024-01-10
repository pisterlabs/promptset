from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import SummaryIndex
from llama_index.response_synthesizers import ResponseMode

import logging
import sys
import openai 
import os

def enable_debug():
    openai.log = "debug"

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def get_documents():
    raw_data = "/root/llm_exam/llama_index/examples/paul_graham_essay/data"

    documents = SimpleDirectoryReader(raw_data).load_data()
    return documents


def get_response_in_model(retriever, response_mode=""):
    # default
    query_engine = RetrieverQueryEngine.from_args(retriever, response_mode=response_mode)
    response = query_engine.query("What did the author do growing up?")

    print()
    print(response_mode)
    print(response)
    print()

    return response


if __name__ == "__main__":
    #enable_debug()
    documents = get_documents()
    index = SummaryIndex.from_documents(documents)

    retriever = index.as_retriever()

    get_response_in_model(retriever, response_mode=ResponseMode.REFINE)
    get_response_in_model(retriever, response_mode=ResponseMode.COMPACT)
    get_response_in_model(retriever, response_mode=ResponseMode.TREE_SUMMARIZE)
    get_response_in_model(retriever, response_mode=ResponseMode.SIMPLE_SUMMARIZE)
    get_response_in_model(retriever, response_mode=ResponseMode.NO_TEXT)
    get_response_in_model(retriever, response_mode=ResponseMode.COMPACT_ACCUMULATE)

    

