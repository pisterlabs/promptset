
from llama_index import download_loader, VectorStoreIndex, SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from dotenv import load_dotenv
import openai
import os
import logging

PATH_TO_DOCS = "./resources"


def load_environment_vars() -> dict:
    """Load required environment variables. Raise an exception if any are missing."""

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

    logging.info("Environment variables loaded.")
    return {"OPENAI_API_KEY": api_key}


def load_and_index_data() -> VectorStoreIndex:
    docs = load_data()
    index = index_data(docs)
    return index


def load_data() -> []:

    documents = SimpleDirectoryReader(
        input_dir=PATH_TO_DOCS, recursive=True
    ).load_data()
    print("LOADED DOCUMENTS")
    return documents


def index_data(docs: []) -> VectorStoreIndex:
    """Index Documents"""

    logging.info("Parsing documents into nodes...")
    parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=32)
    nodes = parser.get_nodes_from_documents(docs)

    logging.info("Indexing nodes...")
    index = VectorStoreIndex(nodes)

    logging.info("Persisting index on ./storage...")
    index.storage_context.persist(persist_dir="./storage")

    logging.info("Data-Knowledge ingestion process is completed (OK)")
    print("Data-Knowledge ingestion process is completed (OK)")
    return index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        env_vars = load_environment_vars()
        openai.api_key = env_vars['OPENAI_API_KEY']

        load_and_index_data()
    except Exception as ex:
        logging.error("Unexpected Error: %s", ex)
        raise ex
