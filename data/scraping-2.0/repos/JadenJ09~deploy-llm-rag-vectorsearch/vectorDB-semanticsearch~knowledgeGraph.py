# %%
import logging
import sys
#
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
# %%
from llama_index import (SimpleDirectoryReader,
                         LLMPredictor,
                         ServiceContext,
                         KnowledgeGraphIndex)
#
from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index.llms import HuggingFaceInferenceAPI
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from llama_index.embeddings import LangchainEmbedding
from pyvis.network import Network
from dotenv import load_dotenv
import os
# %%
load_dotenv()

HF_TOKEN = "HF_TOKEN"
llm = HuggingFaceInferenceAPI(
    model_name="mistralai/Mistral-7B-Instruct-v0.2", token=HF_TOKEN
)
# %%
print(llm)
# %%
embed_model = LangchainEmbedding(
  HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="thenlper/gte-large")
)
# %%
documents = SimpleDirectoryReader("data").load_data()
pdf_documents = [doc for doc in documents if doc.filename.endswith('.pdf')]
print(len(pdf_documents))
# %%
