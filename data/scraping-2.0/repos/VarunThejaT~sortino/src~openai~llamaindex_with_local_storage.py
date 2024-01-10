import openai
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

from llama_index import ServiceContext, download_loader, GPTVectorStoreIndex, SimpleDirectoryReader
from pathlib import Path

UnstructuredReader = download_loader("UnstructuredReader", refresh_cache=True)

loader = UnstructuredReader()
# meta_doc = loader.load_data(file=Path(f'/home/varun/source/sortino/data/meta-20221231.htm'), split_documents=False)
meta_doc = SimpleDirectoryReader(input_files=[f"/home/varun/source/sortino/src/openai/data/meta-20221231.md"]).load_data()
# uber_doc = SimpleDirectoryReader(input_files=[f"data/UBER/UBER_2022.html"]).load_data()

service_context = ServiceContext.from_defaults(chunk_size_limit=256)

########################################
# Indexing is a pretty costly operation, only do once
cur_index = GPTVectorStoreIndex.from_documents(meta_doc, service_context=service_context)
########################################

# reload from disk
from llama_index import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./storage")
# load index
index = load_index_from_storage(storage_context)