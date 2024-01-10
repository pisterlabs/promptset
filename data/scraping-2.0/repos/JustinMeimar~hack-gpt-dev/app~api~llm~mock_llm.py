import os
import time

from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator

"""
mock prompt processing to show where to handle
"""
def process_prompt(prompt):
    if prompt:
        load_dotenv()

        loader = load_all_pages() 
        index = create_index(loader)
        
        return {
            "response": make_query(index, prompt)
        }

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@timer_decorator
def load_all_pages(filetype='.pdf'):

    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, 'data')

    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        loader = PyPDFLoader(file_path) 

        return loader 

@timer_decorator
def create_index(loader):
    index = VectorstoreIndexCreator().from_loaders([loader])

    return index

@timer_decorator
def make_query(index, query): 
    res = index.query(query)

    return res

