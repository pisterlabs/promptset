import logging
import sys
import os
import torch
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.embeddings import LangchainEmbedding
import langchain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from llama_index.callbacks import (
    CallbackManager,
    LlamaDebugHandler
)

# Start script
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

documents = SimpleDirectoryReader(
    input_dir='bd',
    required_exts=[".pdf"]
).load_data()




# https://medium.com/@thakermadhav/build-your-own-rag-with-mistral-7b-and-langchain-97d0c92fa146

'''
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine()
response = query_engine.query("Explain to me the Statement of profit or loss of Ernst & Young Nederland") # 30G RAM y full 24 vCPU
print(response)

# conversation-like
while True:
  query=input()
  response = query_engine.query(query)
  print(response)
'''
