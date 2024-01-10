from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import DataFrameLoader
import pandas as pd


def setup_db(init: bool = False) -> Milvus:
    df = pd.read_csv("data/titles_with_emotions_emojis.csv")
    df = df.dropna()

    loader = DataFrameLoader(df, page_content_column="emotions")

    docs = loader.load()

    embeddings = OpenAIEmbeddings()

    if init:
        vector_db = Milvus.from_documents(
            docs,
            embeddings,
            connection_args={"host": "127.0.0.1", "port": "19530"},
        )
    else:
        vector_db = Milvus(embedding_function=embeddings, connection_args={"host": "127.0.0.1", "port": "19530"})

    return vector_db


