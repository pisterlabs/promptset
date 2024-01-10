import os
import sqlite3
import sys
from uuid import uuid4
from typing import List, Dict

from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pandas as pd

sys.path.append("..")
from common._secrets import DB_PARENT_DIR
from common.utils import chunks


client = QdrantClient("localhost", port=6333)
embeddings = OpenAIEmbeddings()


def create_collection(collection_name: str) -> bool:
    return client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1536, distance=models.Distance.COSINE, on_disk=True
        ),
    )


def get_conn():
    return sqlite3.connect(os.path.join(DB_PARENT_DIR, "knowledge_db.sqlite"))


def create_database_from_json(data: Dict, database_name: str):
    df = pd.DataFrame(data)
    df["uuid"] = df.apply(lambda row: str(uuid4()), axis=1)
    df["collection_name"] = database_name.upper()
    with get_conn() as conn:
        df.to_sql(database_name, conn, if_exists="append", index=False)


def read_table(table_name: str) -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql_query(f"select * from {table_name}", conn)


def query_table_by_uuid(table_name: str, uuid: str) -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql_query(f"select * from {table_name} where uuid = '{uuid}'", conn)


def get_points_for_upsert(df_to_process: pd.DataFrame, col_to_embed: str) -> List[Dict]:
    points = []
    for row in df_to_process.itertuples():
        emb_record = getattr(row, col_to_embed)
        metadata = {
            "uuid": row.uuid,
            "content": emb_record,
            "source": row.collection_name,
        }
        points.append(
            {
                "id": metadata["uuid"],
                "payload": metadata,
                "vector": embeddings.embed_documents([emb_record])[0],
            }
        )
    return points


def upsert_qdrant_collection(points: List[Dict], collection_name: str):
    points_chunks = chunks(points, 100)
    for points in points_chunks:
        ids, vectors, payloads = zip(
            *((point["id"], point["vector"], point["payload"]) for point in points)
        )

        client.upsert(
            collection_name,
            points=models.Batch(ids=ids, payloads=payloads, vectors=vectors),
        )


def query_collection(embedded_query, collection_name, limit=1):
    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=collection_name),
            )
        ]
    )
    search = client.search(
        collection_name,
        query_vector=embedded_query,
        limit=limit,
        query_filter=query_filter,
    )
    return search
