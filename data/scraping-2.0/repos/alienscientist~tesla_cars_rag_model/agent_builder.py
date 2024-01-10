from llama_index.indices.composability import ComposableGraph
from llama_index.indices.keyword_table import SimpleKeywordTableIndex

import json
import logging
import os
from pydantic import BaseModel
import requests

from llama_index import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    SimpleDirectoryReader,
    SummaryIndex,
    ServiceContext,
)
from llama_index.readers.file.flat_reader import FlatReader
from llama_index.node_parser import UnstructuredElementNodeParser, SentenceSplitter
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import RecursiveRetriever
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.callbacks import CallbackManager
from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType
from llama_index.objects import ObjectIndex, SimpleToolNodeMapping
from queue import Queue

from typing import Optional, Dict, Any, List, Tuple

from pathlib import Path
import os
import pickle

from llama_index import (
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
    SimpleDirectoryReader,
    SummaryIndex,
    ServiceContext,
)

def build_document_agents(
    storage_dir: str, docs: Dict[str, Any]
) -> Dict:
    """Build document agents."""
    node_parser = SentenceSplitter()
    llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm)

    # Build agents dictionary
    agents = {}

    # this is for the baseline
    all_nodes = []
    indices = {}
    
    for idx, model in enumerate(docs.keys()):
        nodes = node_parser.get_nodes_from_documents(docs[model])
        all_nodes.extend(nodes)

        if not os.path.exists(f"./{storage_dir}/{model}"):
            # build vector index
            vector_index = VectorStoreIndex(nodes, service_context=service_context)
            vector_index.storage_context.persist(
                persist_dir=f"./{storage_dir}/{model}"
            )
        else:
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(
                    persist_dir=f"./{storage_dir}/{model}"
                ),
                service_context=service_context,
            )

        # build summary index
        summary_index = SummaryIndex(nodes, service_context=service_context)
        # define query engines
        vector_query_engine = vector_index.as_query_engine()
        summary_query_engine = summary_index.as_query_engine()

        indices[model] = VectorStoreIndex.from_documents(
            docs[model],
            service_context=service_context,
            storage_context=vector_index.storage_context,
        )
        
        # define tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        "Useful for questions related to specific aspects of"
                        f" {model} (e.g. the battery, engine,"
                        " spesifications, or more)."
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=(
                        "Useful for any requests that require a holistic summary"
                        f" of EVERYTHING about {model}. For questions about"
                        " more specific sections, please use the vector_tool."
                    ),
                ),
            )
        ]

        # build agent
        function_llm = OpenAI(model="gpt-3.5-turbo")
        agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=function_llm,
            verbose=True,
            system_prompt=f"""\
    You are a specialized agent designed to answer queries about {model}.
    You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
    """,
            
        )

        agents[model] = agent

    return agents, indices, llm

