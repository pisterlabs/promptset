from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import openai
import numpy as np
import sqlite3


class ChatBot:
    def __init__(self, openai_api_key: str, db_path: str):
        openai.api_key = openai_api_key
        self._embeddings = {'embeddings.csv':None}
        self._db_path = db_path # type: ignore
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._qa_pairs = {}

    def _save_embeddings_to_db(self):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        for key, value in self._embeddings.items():
            cursor.execute(
                "INSERT INTO embeddings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value.tolist()))
        conn.commit()
        conn.close()

    def _load_embeddings_from_db(self):
        conn = sqlite3.connect(self._db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM embeddings")
        rows = cursor.fetchall()
        for row in rows:
            self._embeddings[row[0]] = np.array(row[1])
        conn.close()

    def load_embeddings_from_database(self):
        self._load_embeddings_from_db()

    def save_embedding_to_database(self, key: str, value: np.ndarray):
        self._embeddings[key] = value
        self._executor.submit(self._save_embeddings_to_db)

    async def generate_embedding_async(self, text: str) -> np.ndarray:
        response = await self._executor.submit(openai.Embedding.retrieve, text=text, model="gpt-3.5-turbo", max_tokens=512, n=1,stop=None,temperature=0.85)
        embedding = np.array(response["embedding"])
        return embedding

    def get_closest_embedding(self, embedding: np.ndarray) -> Tuple[str, float]:
        similarity_scores = {key: float(np.dot(value, embedding) / (np.linalg.norm(value) * np.linalg.norm(embedding))) for key, value in self._embeddings.items()}
        key, max_score = max(similarity_scores.items(), key=lambda x: x[1])
        return key, max_score

    async def get_answer_async(self, text: str) -> str:
        closest_key, _ = self.get_closest_embedding(await self.generate_embedding_async(text))
        return self._qa_pairs[closest_key]

    class EmbeddingHelper:
        @staticmethod
        def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class EmbeddingData:
    @staticmethod
    async def load_embeddings_async(filename: str) -> Dict[str, np.ndarray]:
        embeddings = {}
        async with open(filename) as f:
            async for line in f:
                key, values = line.split(" ", 1)
                embeddings[key] = np.fromstring(values, sep=" ")
        return embeddings

@staticmethod
async def save_embeddings_async(filename: str, embeddings: Dict[str, np.ndarray]):
    async with open(filename, "w") as f:
        for key, value in embeddings.items():
            f.write(f"{key} {value.tobytes()}\n")