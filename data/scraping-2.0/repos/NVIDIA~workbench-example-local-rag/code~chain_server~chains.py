"""LLM Chains for executing Retrival Augmented Generation."""
import base64
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Generator, List, Optional

import torch
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceTextGenInference


from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from llama_index import (
    LangchainEmbedding,
    Prompt,
    ServiceContext,
    VectorStoreIndex,
    download_loader,
    set_global_service_context,
)
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.llms import LangChainLLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import StreamingResponse, Response
from llama_index.schema import MetadataMode
from llama_index.utils import globals_helper
from llama_index.vector_stores import MilvusVectorStore, SimpleVectorStore

from chain_server import configuration
# from chain_server.trt_llm import TensorRTLLM

if TYPE_CHECKING:
    from llama_index.indices.base_retriever import BaseRetriever
    from llama_index.indices.query.schema import QueryBundle
    from llama_index.schema import NodeWithScore
    from llama_index.types import TokenGen

    from chain_server.configuration_wizard import ConfigWizard

TEXT_SPLITTER_MODEL = "intfloat/e5-large-v2"
TEXT_SPLITTER_CHUNCK_SIZE = 510
TEXT_SPLITTER_CHUNCK_OVERLAP = 200
EMBEDDING_MODEL = "intfloat/e5-large-v2"
DEFAULT_NUM_TOKENS = 50
DEFAULT_MAX_CONTEXT = 800

LLAMA_CHAT_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "You are a helpful, respectful and honest assistant."
    "Always answer as helpfully as possible, while being safe."
    "Please ensure that your responses are positive in nature."
    "<</SYS>>"
    "[/INST] {context_str} </s><s>[INST] {query_str} [/INST]"
)

LLAMA_RAG_TEMPLATE = (
    "<s>[INST] <<SYS>>"
    "Use the following context to answer the user's question. If you don't know the answer,"
    "just say that you don't know, don't try to make up an answer."
    "<</SYS>>"
    "<s>[INST] Context: {context_str} Question: {query_str} Only return the helpful"
    " answer below and nothing else. Helpful answer:[/INST]"
)


class LimitRetrievedNodesLength(BaseNodePostprocessor):
    """Llama Index chain filter to limit token lengths."""

    def postprocess_nodes(
        self, nodes: List["NodeWithScore"], query_bundle: Optional["QueryBundle"] = None
    ) -> List["NodeWithScore"]:
        """Filter function."""
        included_nodes = []
        current_length = 0
        limit = DEFAULT_MAX_CONTEXT

        for node in nodes:
            current_length += len(
                globals_helper.tokenizer(
                    node.node.get_content(metadata_mode=MetadataMode.LLM)
                )
            )
            if current_length > limit:
                break
            included_nodes.append(node)

        return included_nodes


@lru_cache
def get_config() -> "ConfigWizard":
    """Parse the application configuration."""
    config_file = os.environ.get("APP_CONFIG_FILE", "/dev/null")
    config = configuration.AppConfig.from_file(config_file)
    if config:
        return config
    raise RuntimeError("Unable to find configuration.")


@lru_cache
def get_llm() -> LangChainLLM:
    """Create the LLM connection."""
    inference_server_url_local = "http://127.0.0.1:9090/"

    llm_local = HuggingFaceTextGenInference(
        inference_server_url=inference_server_url_local,
        max_new_tokens=100,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.7,
        repetition_penalty=1.03,
        streaming=True
    )

    return LangChainLLM(llm=llm_local)


@lru_cache
def get_embedding_model() -> LangchainEmbedding:
    """Create the embedding model."""
    model_kwargs = {"device": "cpu"}
    device_str = os.environ.get('EMBEDDING_DEVICE', "cuda:1")
    if torch.cuda.is_available():
        model_kwargs["device"] = device_str

    encode_kwargs = {"normalize_embeddings": False}
    hf_embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Load in a specific embedding model
    return LangchainEmbedding(hf_embeddings)


@lru_cache
def get_vector_index() -> VectorStoreIndex:
    """Create the vector db index."""
    config = get_config()
    vector_store = MilvusVectorStore(uri=config.milvus, dim=1024, overwrite=False)
    #vector_store = SimpleVectorStore()
    return VectorStoreIndex.from_vector_store(vector_store)


@lru_cache
def get_doc_retriever(num_nodes: int = 4) -> "BaseRetriever":
    """Create the document retriever."""
    index = get_vector_index()
    return index.as_retriever(similarity_top_k=num_nodes)


@lru_cache
def set_service_context() -> None:
    """Set the global service context."""
    service_context = ServiceContext.from_defaults(
        llm=get_llm(), embed_model=get_embedding_model()
    )
    set_global_service_context(service_context)


def llm_chain(
    context: str, question: str, num_tokens: int
) -> Generator[str, None, None]:
    """Execute a simple LLM chain using the components defined above."""
    set_service_context()
    prompt = LLAMA_CHAT_TEMPLATE.format(context_str=context, query_str=question)
    response = get_llm().complete(prompt, max_new_tokens=num_tokens)

    for i in range(0, len(response.text), 20):
        yield response.text[i:i + 20]

def llm_chain_streaming(
    context: str, question: str, num_tokens: int
) -> Generator[str, None, None]:
    """Execute a simple LLM chain using the components defined above."""
    set_service_context()
    prompt = LLAMA_CHAT_TEMPLATE.format(context_str=context, query_str=question)

    response = get_llm().stream_complete(prompt, max_new_tokens=num_tokens)
    gen_response = (resp.delta for resp in response)
    return gen_response

def rag_chain(prompt: str, num_tokens: int) -> "TokenGen":
    """Execute a Retrieval Augmented Generation chain using the components defined above."""
    set_service_context()
    get_llm().llm.max_new_tokens = num_tokens  # type: ignore
    retriever = get_doc_retriever(num_nodes=4)
    qa_template = Prompt(LLAMA_RAG_TEMPLATE)
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        text_qa_template=qa_template,
        node_postprocessors=[LimitRetrievedNodesLength()],
        streaming=False,
    )
    response = query_engine.query(prompt)

    # Properly handle an empty response
    if isinstance(response, Response):
        for i in range(0, len(response.response), 20):
            yield response.response[i:i + 20]
    return Response([]).response  # type: ignore

def rag_chain_streaming(prompt: str, num_tokens: int) -> "TokenGen":
    """Execute a Retrieval Augmented Generation chain using the components defined above."""
    set_service_context()
    get_llm().llm.max_new_tokens = num_tokens  # type: ignore
    retriever = get_doc_retriever(num_nodes=4)
    qa_template = Prompt(LLAMA_RAG_TEMPLATE)
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        text_qa_template=qa_template,
        node_postprocessors=[LimitRetrievedNodesLength()],
        streaming=True,
    )
    response = query_engine.query(prompt)

    # Properly handle an empty response
    if isinstance(response, StreamingResponse):
        return response.response_gen
    return StreamingResponse([]).response_gen  # type: ignore

def is_base64_encoded(s: str) -> bool:
    """Check if a string is base64 encoded."""
    try:
        # Attempt to decode the string as base64
        decoded_bytes = base64.b64decode(s)
        # Encode the decoded bytes back to a string to check if it's valid
        decoded_str = decoded_bytes.decode("utf-8")
        # If the original string and the decoded string match, it's base64 encoded
        return s == base64.b64encode(decoded_str.encode("utf-8")).decode("utf-8")
    except Exception:  # pylint:disable = broad-exception-caught
        # An exception occurred during decoding, so it's not base64 encoded
        return False


def ingest_docs(data_dir: str, filename: str) -> None:
    """Ingest documents to the VectorDB."""
    unstruct_reader = download_loader("UnstructuredReader")
    loader = unstruct_reader()
    documents = loader.load_data(file=Path(data_dir), split_documents=False)

    encoded_filename = filename[:-4]
    if not is_base64_encoded(encoded_filename):
        encoded_filename = base64.b64encode(encoded_filename.encode("utf-8")).decode(
            "utf-8"
        )

    for document in documents:
        document.metadata = {"filename": encoded_filename}

    index = get_vector_index()

    text_splitter = SentenceTransformersTokenTextSplitter(
        model_name=TEXT_SPLITTER_MODEL,
        chunk_size=TEXT_SPLITTER_CHUNCK_SIZE,
        chunk_overlap=TEXT_SPLITTER_CHUNCK_OVERLAP,
    )
    node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)
    nodes = node_parser.get_nodes_from_documents(documents)
    index.insert_nodes(nodes)
