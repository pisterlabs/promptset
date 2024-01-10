from enum import Enum
from typing import List

from pydantic import BaseModel
from steamship.data import TagKind, TagValueKey
from steamship.data.tags import Tag

from openai.api_spec import validate_model
from openai.request_utils import concurrent_json_posts
from steamship.plugin.outputs.plugin_output import UsageReport, OperationType, OperationUnit


class OpenAIObject(str, Enum):
    LIST = 'list'
    EMBEDDING = 'embedding'


class OpenAIEmbedding(BaseModel):
    object: OpenAIObject  # 'embedding'
    index: int
    embedding: List[float]

    def to_tag(self, model: str) -> Tag:
        return Tag(
            kind=TagKind.EMBEDDING,
            name=model,
            value={
                "service": "openai",
                TagValueKey.VECTOR_VALUE: self.embedding
            },
        )


class OpenAIEmbeddingList(BaseModel):
    object: OpenAIObject  # 'list'
    data: List[OpenAIEmbedding]

    def to_tags(self, model: str) -> List[Tag]:
        return [embedding.to_tag(model) for embedding in self.data]


class OpenAIEmbeddingClient:
    URL = "https://api.openai.com/v1/embeddings"

    def __init__(self, key: str):
        self.key = key

    def request(
            self, model: str, inputs: List[str], **kwargs
    ) -> (List[List[Tag]], List[UsageReport]):
        """Performs an OpenAI request. Throw a SteamshipError in the event of error or empty response."""

        validate_model(model)

        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }

        def items_to_body(items: List[str]):
            return {
                "model": model,
                "input": items
            }

        responses = concurrent_json_posts(self.URL, headers, inputs, 6, items_to_body, "openai")
        usage_reports: List[UsageReport] = []
        tag_lists: List[List[Tag]] = []
        for response in responses:
            obj = OpenAIEmbeddingList.parse_obj(response)
            for embedding in obj.data:
                tag_lists.append([embedding.to_tag(model=model)])
            usage_reports.append(UsageReport(
                operation_unit=OperationUnit.PROMPT_TOKENS,
                operation_type=OperationType.RUN,
                operation_amount=response["usage"]["prompt_tokens"]
            ))
        return tag_lists, usage_reports
