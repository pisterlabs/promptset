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
    embed_model=embed_model,
    callback_manager=callback_manager
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Persist index to disk
index.storage_context.persist("10K-AAPL_AMZN_MRNA_TSLA_index")
# from llama_index import StorageContext, load_index_from_storage
# Rebuild storage context
# storage_context = StorageContext.from_defaults(persist_dir="naval_index")
# Load index from the storage context
# new_index = load_index_from_storage(storage_context)
# new_query_engine = new_index.as_query_engine()
# response = new_query_engine.query("who is this text about?")
# print(response)

query_engine = index.as_query_engine()

# conversation-like
while True:
  query=input()
  response = query_engine.query(query)
  print(response)

