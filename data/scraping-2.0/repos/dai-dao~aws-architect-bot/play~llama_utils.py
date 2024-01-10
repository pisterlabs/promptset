import os
import logging
import sys
from dotenv import load_dotenv
load_dotenv()
import openai
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)
from llama_index import (
    KnowledgeGraphIndex,
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore, SimpleGraphStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.llms import OpenAI



# SECTION: define LLM API Key
openai.api_key = os.environ["OPENAI_API_KEY"]
llm = OpenAI(temperature=0, model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm, chunk_size_limit=512)


def get_nebula_kg_index(persist_dir : str, load_from_disk : bool, documents):
    # SECTION: Nebula configs
    # Set up Nebula from console
    # CREATE SPACE llamaindex(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);
    # USE llamaindex;
    # CREATE TAG entity(name string);
    # CREATE EDGE relationship(relationship string);
    # CREATE TAG INDEX entity_index ON entity(name(256));
    os.environ["NEBULA_USER"] = "root"
    os.environ["NEBULA_PASSWORD"] = "nebula"
    os.environ[
        "NEBULA_ADDRESS"
    ] = "127.0.0.1:9669"  # assumed we have NebulaGraph 3.5.0 or newer installed locally
    space_name = "llamaindex"
    edge_types, rel_prop_names = ["relationship"], [
        "relationship"
    ]  # default, could be omit if create from an empty kg
    tags = ["entity"]  # default, could be omit if create from an empty kg


    # SECTION: Build KG Index for Nebula
    graph_store = NebulaGraphStore(
        space_name=space_name,
        edge_types=edge_types,
        rel_prop_names=rel_prop_names,
        tags=tags,
    )
    if load_from_disk:
        storage_context = StorageContext.from_defaults(graph_store=graph_store, persist_dir = persist_dir)
        kg_index = load_index_from_storage(storage_context = storage_context, service_context = service_context)
    
    else:
        storage_context = StorageContext.from_defaults(graph_store=graph_store)
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=storage_context,
            max_triplets_per_chunk=10,
            space_name=space_name,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            include_embeddings=True,
        )
        # Store to disk
        kg_index.storage_context.persist(persist_dir=persist_dir)


    return kg_index


def get_vector_index(persist_dir : str, load_from_disk : bool, documents):
    if load_from_disk:
        storage_context = StorageContext.from_defaults(persist_dir = persist_dir)
        vector_index = load_index_from_storage(storage_context = storage_context, service_context = service_context)
    else:
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store = vector_store)
        vector_index = VectorStoreIndex.from_documents(documents, storage_context = storage_context, service_context = service_context)
        vector_index.storage_context.persist(persist_dir = persist_dir)
    
    return vector_index

