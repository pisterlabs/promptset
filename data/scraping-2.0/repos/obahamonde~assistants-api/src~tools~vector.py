"""Pinecone client module."""
from __future__ import annotations

from os import environ
from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union, cast
from uuid import uuid4

from glob_utils import APIClient, robust  # type: ignore
from openai import AsyncOpenAI
from pydantic import BaseModel, Field  # pylint: disable=E0611

Value: TypeAlias = Union[str, int, float, bool, list[str]]
Filter: TypeAlias = Literal["$eq", "$ne", "$lt", "$lte", "$gt", "$gte", "$in", "$nin"]
AndOr: TypeAlias = Literal["$and", "$or"]
Query: TypeAlias = Union[
    dict[str, Union[Value, "Query", list["Query"], list[Value]]],
    dict[Filter, Value],
    dict[AndOr, list["Query"]],
]
Vector = List[float]
MetaData = Dict[str, Value]


class QueryBuilder(object):
    """Query builder for Pinecone Query API with MongoDB-like syntax."""

    def __init__(self, field: str = None, query: Query = None):  # type: ignore
        self.field = field
        self.query = query if query else {}

    def __repr__(self) -> str:
        return f"{self.query}"

    def __call__(self, field_name: str) -> QueryBuilder:
        return QueryBuilder(field_name)

    def __and__(self, other: QueryBuilder) -> QueryBuilder:
        return QueryBuilder(query={"$and": [self.query, other.query]})

    def __or__(self, other: QueryBuilder) -> QueryBuilder:
        return QueryBuilder(query={"$or": [self.query, other.query]})

    def __eq__(self, value: Value) -> QueryBuilder:  # type: ignore
        return QueryBuilder(query={self.field: {"$eq": value}})

    def __ne__(self, value: Value) -> QueryBuilder:  # type: ignore
        return QueryBuilder(query={self.field: {"$ne": value}})

    def __lt__(self, value: Value) -> QueryBuilder:
        return QueryBuilder(query={self.field: {"$lt": value}})

    def __le__(self, value: Value) -> QueryBuilder:
        return QueryBuilder(query={self.field: {"$lte": value}})

    def __gt__(self, value: Value) -> QueryBuilder:
        return QueryBuilder(query={self.field: {"$gt": value}})

    def __ge__(self, value: Value) -> QueryBuilder:
        return QueryBuilder(query={self.field: {"$gte": value}})

    def in_(self, values: List[Value]) -> QueryBuilder:
        """MongoDB-like syntax for $in operator."""
        return QueryBuilder(query={self.field: {"$in": values}})

    def nin_(self, values: List[Value]) -> QueryBuilder:
        """MongoDB-like syntax for $nin operator."""
        return QueryBuilder(query={self.field: {"$nin": values}})


class UpsertRequest(BaseModel):
    """
    Represents an upsert request.

    Attributes:
            id (str): The ID of the request.
            values (Vector): The values to be upserted.
            metadata (MetaData): The metadata associated with the request.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    values: Vector = Field(...)
    metadata: MetaData = Field(...)


class Embedding(BaseModel):
    """
    Represents an embedding with values and metadata.

    Attributes:
            values (Vector): The vector values of the embedding.
            metadata (MetaData): The metadata associated with the embedding.
    """

    values: Vector = Field(...)
    metadata: MetaData = Field(...)


class QueryRequest(BaseModel):
    """
    Represents a query request.

    Attributes:
            topK (int): The number of results to retrieve (default is 10).
            filter (Dict[str, Any]): The filter to apply to the query.
            includeMetadata (bool): Whether to include metadata in the results (default is True).
            vector (Vector): The vector to use for the query.
    """

    topK: int = Field(default=10)
    filter: Dict[str, Any] = Field(...)
    includeMetadata: bool = Field(default=True)
    vector: Vector = Field(...)


class QueryMatch(BaseModel):
    """
    Represents a query match.

    Attributes:
            id (str): The ID of the match.
            score (float): The score of the match.
            metadata (MetaData): The metadata associated with the match.
    """

    id: str = Field(...)
    score: float = Field(...)
    metadata: MetaData = Field(...)


class QueryModel(BaseModel):
    """
    Represents a query model.

    Attributes:
            matches (List[QueryMatch]): A list of query matches.
    """

    matches: List[QueryMatch] = Field(...)


class QueryResponse(BaseModel):
    """
    Represents a response to a query.

    Attributes:
            score (float): The score of the response.
            text (str): The text of the response.
    """

    id: str = Field(...)
    score: float = Field(...)
    text: str = Field(...)


class UpsertResponse(BaseModel):
    """
    Represents the response of an upsert operation.

    Attributes:
            upsertedCount (int): The number of documents upserted.
    """

    upsertedCount: int = Field(...)


class PineDantic(APIClient):
    """
    Pinecone class represents a client for interacting with the Pinecone service.

    Attributes:
            base_url (str): The base URL of the Pinecone service.
            headers (dict[str, str]): The headers to be included in the API requests.
            namespace (str): The namespace to be used for the operations (default is "default").
    """

    base_url: str = Field(default=environ["PINECONE_URL"])
    headers: dict[str, str] = Field(default={"api-key": environ["PINECONE_API_KEY"]})
    namespace: str = Field(...)

    @property
    def builder(self) -> QueryBuilder:
        """
        Returns the QueryBuilder instance for interacting with the database.

        Returns:
                QueryBuilder: The QueryBuilder instance.
        """
        return QueryBuilder()

    @robust
    async def _upsert(self, *, vectors: List[Vector], metadata: List[MetaData]):
        payload = {
            "vectors": [
                UpsertRequest(values=vector, metadata=meta).dict()
                for vector, meta in zip(vectors, metadata)
            ]
        }
        response = await self.post("/vectors/upsert", json=payload)
        return UpsertResponse(**response).upsertedCount

    @robust
    async def _query(
        self,
        *,
        vector: Vector,
        expr: Query,
        includeMetadata: bool = True,
        topK: int = 10,
    ) -> List[QueryResponse]:
        payload = QueryRequest(
            filter=cast(Dict[str, Any], expr),
            vector=vector,
            includeMetadata=includeMetadata,
            topK=topK,
        ).dict()
        response = await self.post("/query", json=payload)
        data = QueryModel(**response)
        return [
            QueryResponse(
                score=match.score,
                text=str(match.metadata.get("text", "")).strip(),
                id=match.id,
            )
            for match in data.matches
        ]

    @property
    def ai(self):
        """
        Returns an instance of the AsyncOpenAI class.
        """
        return AsyncOpenAI()

    # async def encode(self, *, text: str | list[str]) -> list[list[float]]:
    #     if isinstance(text, str):
    #         chunked_text = [text[i : i + 256] for i in range(0, len(text), 100)]
    #     else:
    #         chunked_text = [t[i : i + 256] for t in text for i in range(0, len(t), 100)]

    #     response = await self.ai.embeddings.create(
    #         input=chunked_text, model="text-embedding-ada-002"
    #     )
    #     return [r.embedding for r in response.data]
    @robust
    async def encode(self, *, text: str | list[str]) -> list[list[float]]:
        response = await self.ai.embeddings.create(
            input=text, model="text-embedding-ada-002"
        )
        return [r.embedding for r in response.data]

    @robust
    async def query(
        self, *, text: str, expr: Optional[Query] = None, topK: int = 10
    ) -> List[QueryResponse]:
        """
        Queries the Pinecone service with the given text and returns the matching results.

        Args:
                text (str): The text to be queried.
                expr (Optional[Query]): The query expression to filter the results (default is None).
                topK (int): The maximum number of results to return (default is 10).

        Returns:
                List[QueryResponse]: A list of QueryResponse objects representing the matching results.
        """
        if expr is None:
            expr = (self.builder("namespace") == self.namespace).query
        vectors = await self.encode(text=text)
        assert len(vectors) == 1, "Only one text is allowed"
        return await self._query(vector=vectors[0], expr=expr, topK=topK)

    @robust
    async def upsert(self, *, text: str | list[str]):
        """
        Upserts the given text or list of texts into the Pinecone service.

        Args:
                text (str | list[str]): The text or list of texts to be upserted.

        Returns:
                int: The number of upserted items.
        """
        embeddings = await self.encode(text=text)
        await self._upsert(
            vectors=embeddings, metadata=[{"text": text, "namespace": self.namespace}]
        )
