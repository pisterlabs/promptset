from dotenv import load_dotenv
import os
import yaml
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
import openai
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct

from utils import get_text_embedding

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

qdrant_config = yaml.safe_load(open("config.yaml", "r"))['qdrant']
QDRANT_URL = qdrant_config["url"]
EMBEDDING_DIM = qdrant_config["embedding_dim"]


def langchain_setup_docs_vecdb():
    """
    Using Langchain framework to split document, embedding text and store in Vector DB
    """
    loader = DirectoryLoader('./doc/document_txt/', glob="**/*.txt", show_progress=True)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n"]
    )
    docs = text_splitter.split_documents(raw_documents)
    collection_name = qdrant_config['document_collection_name']
    Qdrant.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        url=QDRANT_URL,
        collection_name=collection_name,
        force_recreate=True,
    )


def setup_charts_vecdb() -> None:
    """
    Setup document chart with embedding keyword and store in Vector DB
    :return: None
    """
    directory_path = 'document_chart'
    file_names = os.listdir(directory_path)  # 列出目录下的所有文件和子目录
    file_names = [file for file in file_names if os.path.isfile(os.path.join(directory_path, file))]

    to_vecdb_data = []
    for file_name in file_names:
        vector_id = str(uuid.uuid4())
        code = "-".join(file_name.split("-")[:-1])
        dir = f"/home/ubuntu/intflex/document_chart/{file_name}"
        vector = get_text_embedding(file_name.split("-")[-1].split(".")[0])
        to_vecdb_data.append((vector_id, vector, {"code": code, "dir": dir}))

    qdrant_client = QdrantClient(url=QDRANT_URL)
    collection_name = qdrant_config["document_chart_collection_name"]

    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM,
            distance=models.Distance.COSINE
        ),
    )

    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point[0],
                vector=point[1],
                payload=point[2]
            )
            for point in to_vecdb_data
        ]
    )


if __name__ == "__main__":
    langchain_setup_docs_vecdb()
    # setup_charts_vecdb()
