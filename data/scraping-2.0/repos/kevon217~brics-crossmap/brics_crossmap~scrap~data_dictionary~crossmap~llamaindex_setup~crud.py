import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import pickle
from pathlib import Path

from llama_index import (
    LangchainEmbedding,
    Document,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    set_global_service_context,
)
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from brics_crossmap.data_dictionary.crossmap.setup_llamaindex import (
    setup_index_logger,
    log,
    copy_log,
)
from brics_crossmap.utils import helper

cfg = helper.compose_config(
    config_path="../configs/",
    config_name="config",
    overrides=[],
)

storage_path_root = "C:/Users/armengolkm/Desktop/VS_Code_Projects/BRICS/brics-tools/brics_tools/brics_dd/storage/fitbir/test"
storage_path_root = cfg.indices.index.collections.storage_path_root
storage_path_indices = {}
setup_index_logger.info("Initializing local vector store")
client = chromadb.PersistentClient(path=storage_path_root)
service_context = ServiceContext.from_defaults(llm=None, embed_model=embed_model)
set_global_service_context(service_context)
indices = {}
col = "title"
chroma_collection = client.get_or_create_collection(col)


collection = chroma_collection.get()
ids = collection["ids"]
docs = index_docs[col]
metadatas = []
for d in docs:
    metadatas.append(d.metadata)

doc_to_update = chroma_collection.get(limit=1)
doc_to_update["metadatas"][0] = {**doc_to_update["metadatas"][0], **metadatas[0]}
ids = doc_to_update["ids"]


chroma_collection.update(ids=[ids[0]], metadatas=[metadatas[0]])
updated_doc = chroma_collection.get(limit=1)
print(updated_doc["metadatas"][0])

# delete the last document
print("count before", chroma_collection.count())
chroma_collection.delete(ids=[doc_to_update["ids"][0]])
print("count after", chroma_collection.count())
