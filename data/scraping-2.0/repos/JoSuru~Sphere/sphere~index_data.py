# Import necessary stuff
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # splitting text into chunks
from langchain.vectorstores import Chroma

from sphere.utils import embedding_model


def main(docs):
    # by using RecursiveCharacterTextSplitter we try to split text by chunk size
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
    texts = text_splitter.split_documents(docs)
    print(texts)

    # embedding will help you to create vector space out of your text
    embeddings = embedding_model()

    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk
    persist_directory = "db"

    vectordb = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=persist_directory
    )
    vectordb.persist()
    vectordb = None
