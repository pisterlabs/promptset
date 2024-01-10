import logging
import os, sys
from typing import Any, List
import chromadb
import dataclasses
from utils import get_langchain_docs_url
from llama_index import (
    Document,
    StorageContext,
    ServiceContext,
    VectorStoreIndex,
    LangchainEmbedding,
)
from llama_index.embeddings import OpenAIEmbedding
from llama_index.vector_stores import ChromaVectorStore, PineconeVectorStore
from llama_index.node_parser import SimpleNodeParser
from langchain.embeddings import OpenAIEmbeddings
from custom_types import Source, SourceType
from crawler import WebpageCrawler, SourceType
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# CHROMA
chroma_config = {"client": chromadb.PersistentClient()}
chroma_collection = chroma_config["client"].get_or_create_collection("official")
chroma_vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# PINECONE
pinecone_config = {"environment": "us-west1-gcp-free"}
pinecone_vector_stores = {
    "official": PineconeVectorStore(
        index_name="official",
        environment=pinecone_config["environment"],
        namespace="dev",
    )
}

embed_model = LangchainEmbedding(OpenAIEmbeddings())


def get_urls(sources: List[Source]):
    return [s.url for s in sources]


def get_contents(sources: List[Source]):
    return [s.content for s in sources]


def get_documents(sources: List[Source]):
    return [
        Document(text=s.content, embedding=embed_model.get_text_embedding(s.content))
        for s in sources
    ]


def get_nodes(documents: List[Document]):
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)


def get_metadatas(sources: List[Source]):
    return [s.metadata for s in sources]


def create_index(vector_store, sources: List[Source] = []):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    documents = get_documents(sources)
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
    )
    return index


def get_index(vector_store):
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


def create_official_langchain_index(vector_store):
    langchain_paths = get_langchain_docs_url()
    errored = []
    urls = [*langchain_paths]

    sources = []
    for url in tqdm(urls):
        print(f"Scraping {url}...")
        crawler = WebpageCrawler(
            source_type=SourceType.Official, use_unstructured=False
        )
        try:
            sources.append(crawler.generate_row(url))
        except Exception as e:
            errored.append(url)
            print(f"Error on {url}, {e}")

    index = create_index(vector_store, sources)
    return index


if __name__ == "__main__":
    create_official_langchain_index(pinecone_vector_stores["official"])
