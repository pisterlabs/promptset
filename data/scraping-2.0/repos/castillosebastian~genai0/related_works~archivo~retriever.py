import os
import time
import logging
import configparser
# import sys
# import requests
# import torch
# import langchain
# import sentence_transformers
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding
from src.utils import *

# Configure logging to file
filename = os.path.basename(__file__)
log_file = os.path.join('logs', "retriever.log")
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format=f"%(asctime)s - {filename} - %(levelname)s - %(message)s")
# Read config and set working directory
config = configparser.ConfigParser()
config.read("src/config.ini")
home_dir = config["main"]["HOME_DIR"]
os.chdir(home_dir)


def main():
    
    # Log start time
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info(f"-START: {start_time}")
    
    # Get data
    dirpath = 'data/'
    filename = 'ey.pdf'
    url = 'https://assets.ey.com/content/dam/ey-sites/ey-com/nl_nl/topics/jaarverslag/downloads-pdfs/2022-2023/ey-nl-financial-statements-2023-en.pdf'
    get_pdf_from_url(url)
    #
    documents = SimpleDirectoryReader(
        input_files=[dirpath+filename]
    ).load_data()

    documents = SimpleDirectoryReader(
        input_files=[filename]
    ).load_data()

    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url='https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf',
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": -1},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

    embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="thenlper/gte-large")
    )

    service_context = ServiceContext.from_defaults(
        chunk_size=256,
        llm=llm,
        embed_model=embed_model
    )

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # Log end time    
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info(f"END: {end_time}")
        
    return index
    

if __name__ == "__main__":
    main()
    query_engine = index.as_query_engine()
    # conversation-like
    while True:
        query=input()
        response = query_engine.query(query)
        print(response)
