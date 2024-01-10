import streamlit as st

from typing import List

from openai import ChatCompletion

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models


from src.config import OPENAI_API_KEY, QDRANT_HOST


@st.cache_resource
def connect_to_vectorstore():
    client = QdrantClient(host=QDRANT_HOST, port=6333, path=":memory:")
    try:
        client.get_collection("notion_streamlit")
    except Exception as e:
        print(e)
        client.recreate_collection(
            collection_name="notion_streamlit",
            vectors_config=models.VectorParams(
                size=1536, distance=models.Distance.COSINE
            ),
        )
    return client


def load_data_into_vectorstore(client, docs: List[str]):
    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, client=ChatCompletion
    )
    qdrant_client = Qdrant(
        client=client,
        collection_name="notion_streamlit",
        embedding_function=embeddings.embed_query,
    )
    ids = qdrant_client.add_texts(docs)
    return ids
