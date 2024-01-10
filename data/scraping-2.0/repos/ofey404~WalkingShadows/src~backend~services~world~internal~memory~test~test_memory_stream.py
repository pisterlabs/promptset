import unittest
import uuid

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores.redis import Redis
from libs.testx import REDIS_CONTAINER_VERSION
from redis import Redis as RedisClient
from services.world.internal.memory.memory_stream import MemoryStreamRetriever
from testcontainers.redis import RedisContainer


class TestMemoryStream(unittest.TestCase):
    redis: RedisContainer
    redis_url: str

    @classmethod
    def setUpClass(cls):
        cls.redis = RedisContainer(REDIS_CONTAINER_VERSION)
        cls.redis.start()

        ip = cls.redis.get_container_host_ip()
        port = cls.redis.get_exposed_port(cls.redis.port_to_expose)

        cls.redis_url = f"redis://{ip}:{port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.redis.stop()

    @property
    def ts(self):
        print(f"ts getter, ts = {self._ts}\n")
        return self._ts

    @ts.setter
    def ts(self, value):
        self._ts = value
        print(f"ts setter, ts = {self._ts}\n")

    def setUp(self) -> None:
        self._ts = 0

        embedding = OpenAIEmbeddings()
        self.vectorstore = Redis(
            redis_url=self.redis_url,
            index_name=uuid.uuid4().hex,
            embedding_function=embedding.embed_query,
            content_key="content",
            metadata_key="metadata",
            vector_key="content_vector",
        )

        embeddings = embedding.embed_documents(
            [
                "example text",
            ]
        )
        # Create the search index
        self.vectorstore._create_index(dim=len(embeddings[0]))

        self.ms = MemoryStreamRetriever(
            get_timestamp=lambda: self.ts,
            vectorstore=self.vectorstore,
        )

    def test_get_combined_score(self):
        ts, last_access_ts = 10, 0
        document = Document(
            page_content="Test document",
            metadata={"last_accessed_at": last_access_ts},
        )
        relevance = 0.7

        expected_score = (1.0 - self.ms.decay_rate) ** (ts - last_access_ts) + relevance
        score = self.ms._get_combined_score(
            document,
            relevance,
            ts,
        )
        self.assertAlmostEquals(
            expected_score,
            score,
        )

    def test_lifecycle(self):
        embedding_1 = self.ms.add_documents(
            [
                Document(
                    page_content="hello world",
                    metadata={"last_accessed_at": 0},
                )
            ]
        )
        # print(f"embedding_1 = {embedding_1}\n")

        self.ts = 10
        embedding_2 = self.ms.add_documents(
            [
                Document(page_content="hello foo"),
            ]
        )
        # print(f"embedding_2 = {embedding_2}\n")

        documents = self.ms.get_relevant_documents("hello world")
        # print(documents)
        self.assertEqual(documents[0].page_content, "hello world")

        self.assertEqual(documents[1].page_content, "hello foo")
