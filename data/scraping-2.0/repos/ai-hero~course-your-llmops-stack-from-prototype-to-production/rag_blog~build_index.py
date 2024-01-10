"""Build the vector index from the markdown files in the directory."""
import logging
import os
import sys
from glob import glob
from shutil import rmtree

import openai
from dotenv import load_dotenv
from llama_index import ServiceContext, SimpleDirectoryReader, TreeIndex
from llama_index.llms.openai import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Load the .env file
load_dotenv()

# Set up the OpenAI API key
assert os.getenv("OPENAI_API_KEY"), "Please set your OPENAI_API_KEY environment variable."
openai.api_key = os.getenv("OPENAI_API_KEY")

service_context = ServiceContext.from_defaults(llm=OpenAI())


def build_index(data_dir: str, knowledge_base_dir: str) -> None:
    """Build the vector index from the markdown files in the directory."""
    print("Building vector index...")
    input_files = glob(os.path.join(data_dir, "*.md"))
    documents = SimpleDirectoryReader(input_files=input_files).load_data()

    index = TreeIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=knowledge_base_dir)
    print("Done.")


def main() -> None:
    """Build the vector index from the markdown files in the directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    knowledge_base_dir = os.path.join(base_dir, ".kb")
    # Delete Storage Directory
    if os.path.exists(knowledge_base_dir):
        rmtree(knowledge_base_dir)
    data_dir = os.path.join(base_dir, ".content", "blogs")
    build_index(data_dir, knowledge_base_dir)


if __name__ == "__main__":
    main()
