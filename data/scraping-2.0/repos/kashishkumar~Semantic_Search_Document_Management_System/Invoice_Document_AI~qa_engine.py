from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext
from llama_index.llms import OpenAI
import openai
import torch
#openai.api_key = ""

def load_documents(documents_directory):
    documents = SimpleDirectoryReader(documents_directory).load_data()
    return documents
    
def build_index(documents, llm_model=None, temperature = 0.1, chunk_size = 1024):
    if llm_model is not None:
        llm = OpenAI(temperature, model = llm_model)
        service_context = ServiceContext.from_defaults(llm=llm, chunk_size=chunk_size)
        index =  VectorStoreIndex.from_documents(documents, service_context)
    else:
        index = VectorStoreIndex.from_documents(documents)
    return index 

def build_query_engine(documents_directory, llm_model=None):
    print("Building query engine from your documents")
    documents =  load_documents(documents_directory)
    index =  build_index(documents, llm_model=llm_model)
    query_engine = index.as_query_engine(similarity_top_k = 2)
    print("Query engine built")
    return query_engine

def main():
    documents_directory = input('Enter documents directory name: ')
    openai.api_key = input('Enter OpenAI API key: ')
    query_engine = build_query_engine(documents_directory)
    while True:
        query = input('Enter query (or type "exit" to quit): ')
        if query.lower() == 'exit':
            break
        print(query_engine.query(query))

if __name__ == '__main__':
    main()