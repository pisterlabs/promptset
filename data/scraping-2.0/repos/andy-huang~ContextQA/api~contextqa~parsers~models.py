# pylint: disable=E0611
from enum import Enum
from typing import Any, Type

from langchain.vectorstores.base import VectorStore
from pydantic import BaseModel, Field


class SimilarityProcessor(str, Enum):
    LOCAL = "local"
    PINECONE = "pinecone"


class Summary(BaseModel):
    summary: str = Field(description="Summary of the person")
    facts: list[str] = Field(description="Interesting facts about the person")
    ice_breakers: list[str] = Field(description="Topics of interest of the person")


class LLMResult(BaseModel):
    response: str


class LLMRequestBodyBase(BaseModel):
    separator: str = Field(description="Separator to use for the text splitting", default=".")
    chunk_size: int = Field(description="size of each splitted chunk", default=100)
    chunk_overlap: int = 50


class LLMContextQueryRequest(BaseModel):
    question: str
    processor: SimilarityProcessor
    identifier: str


class LLMQueryRequest(BaseModel):
    message: str
    internet_access: bool = False


class LLMQueryRequestBody(LLMRequestBodyBase):
    query: str = Field(description="The query we want the llm to respond", min_length=10)


class LLMQueryDocumentRequestBody(LLMQueryRequestBody):
    similarity_processor: SimilarityProcessor = SimilarityProcessor.LOCAL


class LLMQueryTextRequestBody(LLMQueryRequestBody):
    content: str = Field(description="The whole content of the 'context'", min_length=500)


class VectorStoreParams(BaseModel):
    clazz: Type[VectorStore]
    kwargs: dict[str, Any] = Field(default_factory=lambda: {})
