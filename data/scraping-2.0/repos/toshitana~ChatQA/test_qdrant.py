# pip install pycryptodome
from glob import glob
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client import models

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection_2"

client = QdrantClient(path=QDRANT_PATH)
text = "エマ"
hits = client.search(
    collection_name="my_books",
    query_vector=OpenAIEmbeddings.embed_query(text),
    limit=3,
)

for hit in hits:
    print(hit.payload, "score:", hit.score)