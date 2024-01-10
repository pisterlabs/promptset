""" Note: This isn't part of the API, just used to generate embeddings for lookups. Will be moved out to a separate process """
import json
import os
import re
import shutil
from typing import List

from chromadb.config import Settings
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


load_dotenv()
# logging.basicConfig(level=logging.DEBUG)

ABS_PATH = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(ABS_PATH, "chroma_embeddings")


def clean_text(text: str) -> str:
    # Replace all newline characters with spaces
    # text = text.replace("\n", " ")

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text


def get_documents():
    loader = TextLoader("knowledgebase/advice.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=1024,
        chunk_overlap=256,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    return texts


def init_chromadb():
    if not os.path.exists(DB_DIR):
        os.mkdir(DB_DIR)
    else:
        shutil.rmtree(DB_DIR)

    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False,
    )
    embeddings = OpenAIEmbeddings(client=None)

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )
    documents = []
    for num, doc in enumerate(get_documents()):
        print(f"Processing document {num}/{len(documents)}")
        doc.page_content = clean_text(doc.page_content)
        documents.append(doc)

    vectorstore.add_documents(documents=documents, embedding=embeddings)
    vectorstore.persist()


def query_chromadb(q: str, k: int = 3, threshold: float = 0.5) -> List[str]:
    if not os.path.exists(DB_DIR):
        raise Exception(f"{DB_DIR} does not exist, nothing can be queried")

    client_settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=DB_DIR,
        anonymized_telemetry=False,
    )

    embeddings = OpenAIEmbeddings(client=None)

    vectorstore = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        client_settings=client_settings,
        persist_directory=DB_DIR,
    )
    docs = vectorstore.similarity_search_with_score(query=q, k=k)
    print(docs)
    filtered_docs = [doc[0].page_content for doc in docs if doc[1] < threshold]

    return filtered_docs


async def main():
    init_chromadb()
    docs = query_chromadb("gobababba oijef pweqilfj erujgn pweoij", 3, 0.5)
    jsonable_result = jsonable_encoder(docs)
    print(json.dumps(jsonable_result, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
