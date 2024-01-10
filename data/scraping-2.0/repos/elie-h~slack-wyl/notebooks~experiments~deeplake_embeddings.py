""" Note: This isn't part of the API, just used to generate embeddings for lookups. Will be moved out to a separate process """
import json
import os
import re
import shutil
from typing import List

from chromadb.config import Settings
import deeplake
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.vectorstores import DeepLake
import os

DEEP_LAKE_API_KEY = os.environ.get("DEEP_LAKE_API_KEY")


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


def init_vs():
    documents = []
    for num, doc in enumerate(get_documents()):
        print(f"Processing document {num}/{len(documents)}")
        doc.page_content = clean_text(doc.page_content)
        documents.append(doc)
    embeddings = OpenAIEmbeddings(client=None)
    # ds = deeplake.empty("hub://eliehamouche/wyl", token=DEEP_LAKE_API_KEY)
    # texts = ds.create_tensor('images', htype='image', sample_compression='jpg')
    db = DeepLake(
        dataset_path="hub://eliehamouche/wyl",
        embedding_function=embeddings,
        token=DEEP_LAKE_API_KEY,
    )
    res = db.add_documents(documents)
    print(res)


def query_vs(q: str, k: int = 3, threshold: float = 0.7) -> List[str]:
    embeddings = OpenAIEmbeddings(client=None)
    vector_store = DeepLake(
        dataset_path="hub://eliehamouche/wyl",
        embedding_function=embeddings,
        token=DEEP_LAKE_API_KEY,
    )
    docs = vector_store.similarity_search_with_score(
        query=q, k=k, distance_metric="cos"
    )
    print(docs)
    filtered_docs = [doc[0].page_content for doc in docs if doc[1] > threshold]

    return filtered_docs


async def main():
    # init_vs()
    docs = query_vs(
        "what's the best software engineering language?!",
        3,
        0.75,
    )
    jsonable_result = jsonable_encoder(docs)
    print(json.dumps(jsonable_result, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
