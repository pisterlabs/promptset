import streamlit as st

import os
from dotenv import load_dotenv
import datetime
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    KnowledgeGraphIndex,
)
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
import transformers
import torch
import time
from llama_index import StorageContext, load_index_from_storage
from llama_index.graph_stores import SimpleGraphStore
from llama_index.llms import OpenAI

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")
DB_URL = os.getenv("DB_URL")

CACHE_DIR = os.getenv("CACHE_DIR")
TOKEN = os.getenv("HF_TOKEN")

EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH")
SOURCE_DOCUMENTS_PATH = os.getenv("SOURCE_DOCUMENTS_PATH")
ASSET_MAPPING_PATH = os.getenv("ASSET_MAPPING_PATH")

EXPERIMENT_LOGGER_STRUCTURED = os.getenv("EXPERIMENT_LOGGER_STRUCTURED")
EXPERIMENT_LOGGER_UNSTRUCTURED = os.getenv("EXPERIMENT_LOGGER_UNSTRUCTURED")
EXPERIMENT_LOGGER_AUTO = os.getenv("EXPERIMENT_LOGGER_AUTO")

CHAT_HISTORY_AUTO = os.getenv("CHAT_HISTORY_AUTO")
CHAT_HISTORY_STRUCTURED = os.getenv("CHAT_HISTORY_STRUCTURED")
CHAT_HISTORY_UNSTRUCTURED = os.getenv("CHAT_HISTORY_UNSTRUCTURED")

VECTOR_DB_INDEX = os.getenv("VECTOR_DB_INDEX")
GRAPH_DB_INDEX = os.getenv("GRAPH_DB_INDEX")


@st.cache_resource
def get_llm(model_name, token, cache_dir):
    if model_name.lower() == "openai":
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
        return llm

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_auth_token=token, cache_dir=cache_dir
    )

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_name,
        use_auth_token=token,
        trust_remote_code=True,
        cache_dir=cache_dir,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=500,
        generate_kwargs={"temperature": 0.1},
        tokenizer=tokenizer,
        model_name=model_name,
        device_map="cuda:0",
        model_kwargs={
            "trust_remote_code": True,
            "config": model_config,
            "quantization_config": bnb_config,
            "use_auth_token": token,
            "cache_dir": cache_dir,
        },
    )

    return llm


@st.cache_resource
def get_service_context(model_name, token, cache_dir):
    llm = get_llm(model_name, token, cache_dir)
    if model_name.lower() == "openai":
        service_context = ServiceContext.from_defaults(llm=llm)
    else:
        service_context = ServiceContext.from_defaults(
            llm=llm, embed_model="local:BAAI/bge-small-en"
        )

    return service_context


def load_docs_and_save_index(model_name, service_context):
    reader = SimpleDirectoryReader(
        input_dir=SOURCE_DOCUMENTS_PATH, required_exts=[".txt", ".pdf"], recursive=True
    )

    docs = reader.load_data()

    for file, doc in zip(reader.input_files, docs):
        ticker = file.parts[-3]
        title = file.parts[-1]
        doc.metadata["title"] = title
        doc.metadata["ticker"] = ticker

    final_docs = [
        doc
        for doc in docs
        if "Our engineers are working quickly to resolve the issue" not in doc.text
    ]

    index = VectorStoreIndex.from_documents(final_docs, service_context=service_context)
    index.storage_context.persist(os.path.join(VECTOR_DB_INDEX, model_name))


def get_query_engine(model_name, service_context):
    if not os.path.exists(os.path.join(VECTOR_DB_INDEX, model_name)):
        os.makedirs(os.path.join(VECTOR_DB_INDEX, model_name))

    storage_context = StorageContext.from_defaults(
        persist_dir=os.path.join(VECTOR_DB_INDEX, model_name)
    )

    index = load_index_from_storage(storage_context, service_context=service_context)

    unstructured_query_engine = index.as_query_engine(similarity_top_k=4)
    return unstructured_query_engine


# def load_docs_and_save_graph_index(service_context):

#     reader = SimpleDirectoryReader(input_dir=SOURCE_DOCUMENTS_PATH,
#                                    required_exts=['.txt'],
#                                    recursive=True)

#     docs = reader.load_data()

#     for file, doc in zip(reader.input_files, docs):
#         ticker = file.parts[-3]
#         title = file.parts[-1]
#         doc.metadata['title'] = title
#         doc.metadata['ticker'] = ticker

#     final_docs = [doc for doc in docs if
#                   "Our engineers are working quickly to resolve the issue"
#                   not in doc.text]

#     graph_store = SimpleGraphStore()
#     storage_context = StorageContext.from_defaults(graph_store=graph_store)

#     index = KnowledgeGraphIndex.from_documents(
#         final_docs[:5],
#         max_triplets_per_chunk=2,
#         storage_context=storage_context,
#         service_context=service_context,
#     )
#     index.storage_context.persist(GRAPH_DB_INDEX)

# def get_graph_query_engine(service_context):

#     storage_context = StorageContext.from_defaults(persist_dir=GRAPH_DB_INDEX)
#     index = load_index_from_storage(storage_context, service_context=service_context)

#     unstructured_query_engine = index.as_query_engine(similarity_top_k=4, response_mode="tree_summarize")
#     return unstructured_query_engine
