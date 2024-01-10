import os
import openai
import logging
import sys
from llama_index.node_parser import SimpleNodeParser
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader, 
    LLMPredictor,
    ServiceContext,
    StorageContext
)

# Enable output logging so you can see whats going on behind the scenes from the terminal
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_key = os.environ.get('OPENAI_API_KEY')


def construct_index():
    index = None
    try:
        documents = SimpleDirectoryReader('ingest').load_data()
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist()
    except:
        print("the storage context did not load correctly")
        exit
    return index

construct_index()
