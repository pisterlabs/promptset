import time
from typing import List, Optional

import openai
import psycopg

from ragen.file import Chunk


def generate_prompt(context: List[str], request: Chunk) -> str:
    context_text = "\n".join(context)
    return f"""Answer the question "{request.text}" with the following context:
{context_text}"""


class OpenAIClient:
    def __init__(
        self, model_name: str, api_key: str, api_base: Optional[str] = None
    ) -> None:
        self.model = model_name
        openai.api_key = api_key
        if api_base:
            openai.api_base = api_base

    def chat(self, prompt: str):
        chat = openai.ChatCompletion.create(
            model=self.model,
            stream=True,
            max_tokens=1000,
            temperature=1,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        for result in chat:
            delta = result.choices[0].delta
            print(delta.get("content", ""), end="", flush=True)
        print()

    def embeddings(self, text: str) -> List[float]:
        emb = openai.Embedding.create(
            model=self.model,
            input=text,
        )
        return emb["data"][0]["embedding"]


class PgClient:
    LOAD_EXTENSION = """
CREATE EXTENSION IF NOT EXISTS vectors;
"""
    CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    filename TEXT NOT NULL,
    index integer NOT NULL,
    emb vector({}) NOT NULL,
    tags text[] NOT NULL
);
"""
    CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS vec_search ON chunks USING vectors (emb l2_ops)
WITH (options = "capacity = 2097152");
"""
    INSERT = """
INSERT INTO chunks (text, filename, index, emb, tags) VALUES (%s, %s, %s, %s, %s)
"""
    SEARCH = """
SELECT text, emb <-> %s AS score
FROM chunks
ORDER BY score LIMIT %s;
"""
    SEARCH_WITH_FILTER = """
SELECT text, emb <-> %s AS score
FROM chunks
WHERE %s = ANY(tags)
ORDER BY score LIMIT %s;
"""

    def __init__(
        self, host: str, user: str, password: str, port: int, dim: int
    ) -> None:
        self.params = dict(
            host=host,
            port=port,
            user=user,
            password=password,
        )
        self.dim = dim
        self.init_db()

    def init_db(self):
        t0 = time.perf_counter()
        print("init db table and pgvecto.rs extension ...")
        with psycopg.connect(**self.params) as conn, conn.cursor() as cur:
            cur.execute(self.LOAD_EXTENSION)
            cur.execute(psycopg.sql.SQL(self.CREATE_TABLE).format(self.dim))
            conn.commit()
        print(f"init done in {time.perf_counter() - t0} s")

    def insert_chunk(self, chunk: Chunk):
        with psycopg.connect(**self.params) as conn, conn.cursor() as cur:
            cur.execute(
                self.INSERT,
                (chunk.text, chunk.filename, chunk.index, str(chunk.emb), chunk.tags),
            )
            conn.commit()

    def indexing(self):
        t0 = time.perf_counter()
        print("indexing ...")
        with psycopg.connect(**self.params) as conn, conn.cursor() as cur:
            cur.execute(self.CREATE_INDEX)
            conn.commit()
        print(f"indexing done in {time.perf_counter() - t0} s")

    def retrieve_similar_chunk(self, chunk: Chunk, top_k: int = 5):
        with psycopg.connect(**self.params) as conn, conn.cursor() as cur:
            cur.execute(self.SEARCH, (str(chunk.emb), top_k))
            result = cur.fetchall()
            conn.commit()
        return result

    def retrieve_with_filter(self, chunk: Chunk, tag: str, top_k: int = 5):
        with psycopg.connect(**self.params) as conn, conn.cursor() as cur:
            cur.execute(self.SEARCH_WITH_FILTER, (str(chunk.emb), tag, top_k))
            result = cur.fetchall()
            conn.commit()
        return result
