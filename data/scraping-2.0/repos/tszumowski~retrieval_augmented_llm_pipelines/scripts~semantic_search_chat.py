"""

This script is an example of how to use llama-index library
to have a vector-store index backed chat agent.

The chat agent defaults to ReACT type using gpt-3.5-turbo-16k model.
It does NOT provide sources like query_engine.

See https://gpt-index.readthedocs.io/en/latest/core_modules/query_modules/chat_engines/usage_pattern.html#available-chat-modes.

Setup:

Set OPENAI_API_KEY and PINECONE_API_KEY env variables

Usage:

Basically same as query_semantic_search_llm.py without the --query argument.
Try first with --help

"""
import argparse
import os
import pinecone

from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        description="Script to test the chat engine with vector store index"
    )
    parser.add_argument(
        "--pinecone_index_name",
        type=str,
        required=True,
        help="Name of pinecone index",
    )
    parser.add_argument(
        "--pinecone_env_name",
        type=str,
        required=True,
        help="Name of pinecone environment",
    )
    parser.add_argument(
        "--pinecone_namespace",
        type=str,
        default=None,
        help="Namespace of pinecone index",
    )
    # top_k, max_text_print
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to return, e.g. 5",
    )
    parser.add_argument(
        "--max_text_print",
        type=int,
        default=500,
        help="Max number of characters to print from each returned text, e.g. 1000",
    )
    # add optional language_model, defaulting to gpt-3.5-turbo-16k
    parser.add_argument(
        "--language_model",
        type=str,
        default="gpt-3.5-turbo-16k",
        help="Name of language model to use, e.g. gpt-3.5-turbo-16k",
    )
    args = parser.parse_args()
    pinecone_index_name = args.pinecone_index_name
    pinecone_env_name = args.pinecone_env_name
    namespace = args.pinecone_namespace
    top_k = args.top_k
    max_text_print = args.max_text_print
    language_model = args.language_model

    # Get API Keys from env
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")

    # Initialize Pinecone
    pinecone.init(environment=pinecone_env_name, api_key=pinecone_api_key)
    pinecone_index = pinecone.Index(pinecone_index_name)

    # Initialize vector store
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index, namespace=namespace
    )

    # Build index from existing vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store,  # vector_store_info=metadata_fields
    )

    # Create language model and bind to service context
    gpt_model = OpenAI(temperature=0, model=language_model)
    service_context_gpt = ServiceContext.from_defaults(llm=gpt_model)

    # Create engine
    chat_engine = index.as_chat_engine(
        service_context=service_context_gpt, verbose=True
    )

    # Start interactive chat
    chat_engine.chat_repl()
