from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, ServiceContext, load_index_from_storage
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
import os
import openai
openai.api_key = "OPENAI_API_KEY"

def load_document():
    uploaded_file = None
    # enter code 
    return uploaded_file

def parse_document(uploaded_file):
    # Create test list using uploaded file content
    # What is the data format of uploaded_file?
    text_list = []
    documents = [Document(text=text) for text in text_list]
    parser = SimpleNodeParser().from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    return documents, nodes

def index_data(documents, nodes = None, chunk_size=None):
    # Persist the index to disk
    if nodes is not None:
        if chunk_size is not None:
            service_context = ServiceContext.from_defaults(chunk_size=chunk_size)
            index = VectorStoreIndex(nodes, service_context=service_context, show_progress=True)
        else:
            index = VectorStoreIndex(nodes, show_progress=True)
    else:
        if chunk_size is not None:
            service_context = ServiceContext.from_defaults(chunk_size=chunk_size)
            index = VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=True)
        else:
            index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index

def save_query_engine(index, storage_context, save_index = True):
    index.save(storage_context)
    return None

def build_query_engine(index, streaming = False):
    query_engine = index.as_query_engine(streaming=streaming)
    return query_engine

def output_response(query_engine, query):
    response = query_engine.query(query)
    return response

def show_response(response):
    print(response) 
    
def qa_engine(uploaded_file, query):
    documents, nodes = parse_document(uploaded_file)
    index = index_data(documents, nodes)
    storage_context = StorageContext.from_defaults()
    save_query_engine(index, storage_context, save_index = False)
    query_engine = build_query_engine(index)
    response = output_response(query_engine, query)
    show_response(response)
    

def main():
    uploaded_file = load_document()
    query = "What is the date for importer agreement?"
    qa_engine(uploaded_file, query)