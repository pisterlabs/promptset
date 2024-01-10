from abc import ABC, abstractmethod
from typing import List, Optional


from ..models.models import (
    Query,
    QueryResult,
    QueryWithEmbedding,
)

from ..services import openai


class DataStore(ABC):
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return openai.get_embeddings(texts)

    async def query(self, queries: List[Query], top_k=10) -> List[QueryResult]:
        """
        Takes in a list of queries and filters and returns a list of query results with matching document chunks and scores.
        """
        # get a list of of just the queries from the Query list
        query_texts = [query.query for query in queries]
        query_embeddings = self.get_embeddings(query_texts)
        # hydrate the queries with embeddings
        queries_with_embeddings = [
            QueryWithEmbedding(**query.dict(), embedding=embedding)
            for query, embedding in zip(queries, query_embeddings)
        ]
        return await self._query(queries_with_embeddings, top_k=top_k)

    @abstractmethod
    async def _query(
        self, queries: List[QueryWithEmbedding], top_k=10
    ) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        raise NotImplementedError
