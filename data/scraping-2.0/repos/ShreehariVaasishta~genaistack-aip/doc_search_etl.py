"""
External Deps:

pip install unstructured
pip install "unstructured[pdf]"
"""
import os

from dotenv import load_dotenv
from genai_stack.embedding.langchain import LangchainEmbedding as LCEmbedding
from genai_stack.etl.langchain import LangchainETL as LCETL
from genai_stack.stack.stack import Stack
from genai_stack.vectordb.weaviate_db import Weaviate as WeaviateDB

load_dotenv()

# Initialize embedding component
embedding = LCEmbedding.from_kwargs(
    name="HuggingFaceEmbeddings",
    fields={
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "model_kwargs": {"device": "cpu"},
        "encode_kwargs": {"normalize_embeddings": False},
    },
)
print(">>> Embedding Configured")

# Initialize vectordb component
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
weaviate_db = WeaviateDB.from_kwargs(
    url=WEAVIATE_URL,
    index_name="Testing",
    text_key="test",
    attributes=["source", "page"],
)
print(">>> VectorDB Configured")

# Initialize etl component
etl = LCETL.from_kwargs(
    name="DirectoryLoader",
    fields={
        "path": os.getenv("TARGET_FILES_DIR"),
        "glob": "**/*.pdf",
        "use_multithreading": True,
        "show_progress": True,
    },
)
print(">>> ETL Configured")

# Stack the components
stack = Stack(model=None, embedding=embedding, vectordb=weaviate_db, etl=etl)
print(">>> Stacking of components Done...")

# Search the db
if doc := stack.vectordb.similarity_search("Broadcast Technicians"):
    print([{"content": i.page_content, "page": i.metadata["page"], "path": i.metadata["source"]} for i in doc])
else:
    print("Empty...")
