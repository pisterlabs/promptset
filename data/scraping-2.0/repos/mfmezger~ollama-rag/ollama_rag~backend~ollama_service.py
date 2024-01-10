"""Ollama Backend Service."""
import os

from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import Qdrant
from loguru import logger
from omegaconf import DictConfig
from qdrant_client import QdrantClient
import nltk
from ollama_rag.utils.configuration import load_config

nltk.download('punkt')
load_dotenv()


class OllamaService:
    """Ollama Backend Service."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str):
        """Initialize the Ollama Service."""
        self.cfg = cfg
        self.collection_name = collection_name
        self.embedding = OllamaEmbeddings(base_url=cfg.ollama_embeddings.url, model=cfg.ollama_embeddings.model)
        qdrant_client = QdrantClient(
            cfg.qdrant.url,
            port=cfg.qdrant.port,
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=cfg.qdrant.prefer_grpc,
        )
        self.vector_db = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=self.embedding,
        )
        self.model = Ollama(base_url=cfg.ollama.url, model=cfg.ollama.model)

    def embedd_documents(self, dir: str) -> None:
        """Embedding all pdfs in a directory."""
        loader = DirectoryLoader(dir, glob="*.pdf", loader_cls=PyPDFLoader)
        splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = loader.load_and_split(splitter)
        logger.info(f"Loaded {len(docs)} documents.")
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        self.vector_db.add_texts(texts=texts, metadatas=metadatas)
        logger.info("SUCCESS: Texts embedded.")


if __name__ == "__main__":
    ollama = OllamaService(collection_name="ollama")
    ollama.embedd_documents(dir="tests/resources/")
