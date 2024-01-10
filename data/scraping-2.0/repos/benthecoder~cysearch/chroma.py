import os

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import pandas as pd


def query_collection(collection, query, max_results, dataframe):
    results = collection.query(
        query_texts=query, n_results=max_results, include=["distances"]
    )
    df = pd.DataFrame(
        {
            "id": results["ids"][0],
            "score": results["distances"][0],
            "title": dataframe[dataframe.vector_id.isin(results["ids"][0])]["title"],
            "content": dataframe[dataframe.vector_id.isin(results["ids"][0])]["text"],
        }
    )

    return df


df = pd.read_csv("../data/course_w_embeddings.csv")
df["vector_id"] = df.reset_index().index.astype(str)


persist_directory = "chroma_persistence"  # Directory to store persisted Chroma data.
client = chromadb.Client(
    Settings(
        persist_directory=persist_directory,
        chroma_db_impl="duckdb+parquet",
    )
)


EMBEDDING_MODEL = "text-embedding-ada-002"

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL
)

course_collection = client.create_collection(
    name="course_info", embedding_function=embedding_function
)

course_collection.add(
    ids=df.vector_id.tolist(),
    embeddings=df.embedding.tolist(),
)

query_res = query_collection(
    collection=course_collection,
    query="modern art in Europe",
    max_results=10,
    dataframe=df,
)
query_res.head()
