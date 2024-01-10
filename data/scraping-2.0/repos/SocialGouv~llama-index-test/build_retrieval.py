from llama_index.response.notebook_utils import display_response
from llama_index.query_engine import SubQuestionQueryEngine, RouterQueryEngine
from typing import Callable, Optional
import shutil
from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.retrievers import AutoMergingRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.chat_engine import SimpleChatEngine
import openai
import os
import sys
from llama_index import SimpleDirectoryReader, Document
from llama_docs_bot.markdown_docs_reader import MarkdownDocsReader
from llama_index.schema import MetadataMode
from llama_index.node_parser import HierarchicalNodeParser, SimpleNodeParser, get_leaf_nodes
from llama_index.llms import OpenAI
from llama_index import ServiceContext, set_global_service_context
from llama_index import QueryBundle
from llama_index.utils import globals_helper
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

if 'OPENAI_API_KEY' not in os.environ:
    raise Exception("OPENAI_API_KEY environment variable not set")

openai.api_key = os.environ['OPENAI_API_KEY']

def load_service_context(model="gpt-3.5-turbo-16k", max_tokens=512, temperature=0.1):
    # Use local embeddings + gpt-3.5-turbo-16k
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(model, max_tokens=max_tokens, temperature=temperature),
        # embed_model="local:BAAI/bge-base-en"
    )
    return service_context


def load_markdown_docs(filepath, hierarchical=True):
    """Load markdown docs from a directory, excluding all other file types."""
    loader = SimpleDirectoryReader(
        input_dir=filepath,
        required_exts=[".md"],
        file_extractor={".md": MarkdownDocsReader()},
        recursive=True
    )

    documents = loader.load_data()

    if hierarchical:
        # combine all documents into one
        documents = [
            Document(text="\n\n".join(
                    document.get_content(metadata_mode=MetadataMode.ALL)
                    for document in documents
                )
            )
        ]

        # chunk into 3 levels
        # majority means 2/3 are retrieved before using the parent
        large_chunk_size = 1536
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[
                large_chunk_size,
                large_chunk_size // 3,
            ]
        )

        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes, get_leaf_nodes(nodes)
    else:
        node_parser = SimpleNodeParser.from_defaults()
        nodes = node_parser.get_nodes_from_documents(documents)
        return nodes


def get_query_engine_tool(directory, description, hierarchical=True, postprocessors=None):
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=f"./data_{os.path.basename(directory)}"
        )
        index = load_index_from_storage(storage_context)

        if hierarchical:
            retriever = AutoMergingRetriever(
                index.as_retriever(similarity_top_k=6),
                storage_context=storage_context
            )
        else:
            retriever = index.as_retriever(similarity_top_k=12)
    except:
        if hierarchical:
            nodes, leaf_nodes = load_markdown_docs(
                directory, hierarchical=hierarchical)

            docstore = SimpleDocumentStore()
            docstore.add_documents(nodes)
            storage_context = StorageContext.from_defaults(docstore=docstore)

            index = VectorStoreIndex(
                leaf_nodes, storage_context=storage_context)
            index.storage_context.persist(
                persist_dir=f"./data_{os.path.basename(directory)}")

            retriever = AutoMergingRetriever(
                index.as_retriever(similarity_top_k=12),
                storage_context=storage_context
            )

        else:
            nodes = load_markdown_docs(directory, hierarchical=hierarchical)
            index = VectorStoreIndex(nodes)
            index.storage_context.persist(
                persist_dir=f"./data_{os.path.basename(directory)}")

            retriever = index.as_retriever(similarity_top_k=12)

    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        node_postprocessors=postprocessors or [],
    )

    return QueryEngineTool(query_engine=query_engine, metadata=ToolMetadata(name=directory, description=description))

class LimitRetrievedNodesLength:

    def __init__(self, limit: int = 3000, tokenizer: Optional[Callable] = None):
        self._tokenizer = tokenizer or globals_helper.tokenizer
        self.limit = limit

    def postprocess_nodes(self, nodes, query_bundle):
        included_nodes = []
        current_length = 0

        for node in nodes:
            current_length += len(self._tokenizer(
                node.node.get_content(metadata_mode=MetadataMode.LLM)))
            if current_length > self.limit:
                break
            included_nodes.append(node)

        return included_nodes


def build_final_query_engine(service_context):
    # Here we define the directories we want to index, as well as a description for each
    # NOTE: these descriptions are hand-written based on my understanding. We could have also
    # used an LLM to write these, maybe a future experiment.
    docs_directories = {
        # "./docs/community": "Useful for information on community integrations with other libraries, vector dbs, and frameworks.",
        # "./docs/core_modules/agent_modules": "Useful for information on data agents and tools for data agents.",
        # "./docs/core_modules/data_modules": "Useful for information on data, storage, indexing, and data processing modules.",
        # "./docs/core_modules/model_modules": "Useful for information on LLMs, embedding models, and prompts.",
        # "./docs/core_modules/query_modules": "Useful for information on various query engines and retrievers, and anything related to querying data.",
        # "./docs/core_modules/supporting_modules": "Useful for information on supporting modules, like callbacks, evaluators, and other supporting modules.",
        # "./docs/getting_started": "Useful for information on getting started with LlamaIndex.",
        # "./docs/development": "Useful for information on contributing to LlamaIndex development.",
        "./content/standup-fabrique": "Pour consulter l'actualité et les chiffres d'une startup.",
        "./content/support-sre-fabrique": "Pour les questions techniques et développement et déploiement."
    }

    # Build query engine tools
    query_engine_tools = [
        get_query_engine_tool(
            directory,
            description,
            hierarchical=True,
            postprocessors=[LimitRetrievedNodesLength(limit=3000)]
        ) for directory, description in docs_directories.items()
    ]

    # build top-level router -- this will route to multiple sub-indexes and aggregate results
    # query_engine = SubQuestionQueryEngine.from_defaults(
    #     query_engine_tools=query_engine_tools,
    #     service_context=service_context,
    #     verbose=False
    # )

    query_engine = RouterQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        service_context=service_context,
        select_multi=True
    )

    return query_engine

if __name__ == "__main__":
    service_context = load_service_context()
    set_global_service_context(service_context)
    query_engine = build_final_query_engine(service_context)

    response = query_engine.query("Quelle lib utiliser pour accéder à ma base de données ?")

    display_response(response)
