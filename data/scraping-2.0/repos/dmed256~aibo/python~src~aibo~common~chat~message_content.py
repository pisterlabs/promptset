from __future__ import annotations

import abc
import json
import typing
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Literal, Optional, Self, Union
from uuid import UUID

import openai
from pydantic import BaseModel, Field, TypeAdapter

from aibo.common.openai import CompletionError, OpenAIContent, OpenAIModel
from aibo.common.types import JsonValue

if TYPE_CHECKING:
    from aibo.core.chat import Conversation

__all__ = [
    "CompletionErrorContent",
    "FunctionRequestContent",
    "FunctionResponseContent",
    "FunctionResponseStatus",
    "FunctionResponseErrorType",
    "TextMessageContent",
    "ImageMessageContent",
    "stringify_message_contents",
    "MessageContent",
    "MessageContentAdapter",
    "MessageContents",
    "MessageContentsAdapter",
]


class BaseMessageContent(BaseModel, abc.ABC):
    """
    Each message content must be derived from BaseMessageContent
    """

    # Unique to each message content type
    kind: str

    @abc.abstractmethod
    async def to_openai(self, *, openai_model: OpenAIModel) -> Optional[OpenAIContent]:
        ...

    @abc.abstractmethod
    def __str__(self) -> str:
        ...


class TextMessageContent(BaseMessageContent):
    """
    Regular text content
    """

    kind: Literal["text"] = "text"
    text: str

    async def to_openai(self, *, openai_model: OpenAIModel) -> OpenAIContent:
        return {
            "type": "text",
            "text": self.text,
        }

    def __str__(self) -> str:
        return self.text


class ImageMessageContent(BaseMessageContent):
    """
    Image content
    """

    kind: Literal["image"] = "image"
    image_id: UUID

    async def to_openai(self, *, openai_model: OpenAIModel) -> OpenAIContent:
        from aibo.db.models import ImageModel

        if "image" not in openai_model.modalities:
            return {
                "type": "text",
                "text": "[Image placeholder]",
            }

        image = await ImageModel.by_id(self.image_id)
        if not image:
            return {
                "type": "text",
                "text": "[Image missing]",
            }

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{image.format};base64,{image.contents_b64}"
            },
        }

    def __str__(self) -> str:
        return f"[Image:{self.image_id}]"


class CompletionErrorContent(BaseMessageContent):
    """
    Error from sampling a message completion
    """

    kind: Literal["completion_error"] = "completion_error"
    error_type: CompletionError.ErrorType
    text: str

    @classmethod
    def from_error(cls, error: CompletionError) -> Self:
        return cls(
            error_type=error.error_type,
            text=error.text,
        )

    @classmethod
    def from_openai(cls, error: openai.OpenAIError) -> Self:
        return cls.from_error(CompletionError.from_openai(error))

    async def to_openai(self, *, openai_model: OpenAIModel) -> None:
        return None

    def __str__(self) -> str:
        return f"Error {self.error_type}: {self.text}"


class FunctionRequestContent(BaseMessageContent):
    """
    Message request to a function
    """

    kind: Literal["function_request"] = "function_request"
    package: str = Field(
        title="package", description="This is the function package name"
    )
    function: str = Field(title="function", description="This is the function's name")
    text: str

    @property
    def arguments(self) -> dict[str, Any]:
        return typing.cast(dict[str, Any], json.loads(self.text))

    def get_openai_function_name(self) -> str:
        from aibo.core.package import Package

        return Package.get_openai_function_name(
            package=self.package,
            function=self.function,
        )

    def get_openai_function_call(self) -> dict[str, Any]:
        return {
            "name": self.get_openai_function_name(),
            "arguments": json.dumps(self.arguments, indent=2),
        }

    async def to_openai(self, *, openai_model: OpenAIModel) -> OpenAIContent:
        return {
            "type": "text",
            "text": self.text,
        }

    def __str__(self) -> str:
        return f"{self.package}.{self.function}({self.text})"


class FunctionResponseStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class FunctionResponseErrorType(StrEnum):
    UNKNOWN = "unknown"
    INVALID_PACKAGE_NAME = "invalid_package_name"
    INVALID_FUNCTION_NAME = "invalid_function_name"
    INVALID_REQUEST = "invalid_request"
    FUNCTION_ERROR = "function_error"


class FunctionResponseContent(BaseMessageContent):
    """
    Response from a function
    """

    kind: Literal["function_response"] = "function_response"
    package: str = Field(
        title="package", description="This is the function package name"
    )
    function: str = Field(title="function", description="This is the function's name")
    status: FunctionResponseStatus = Field(
        title="status",
        description='Returns if the action was successfully processed. Possible values: "success", "error"]',
    )
    error_type: Optional[FunctionResponseErrorType]
    error_message: Optional[str]
    arguments: dict[str, Any]
    response: JsonValue

    def get_openai_function_name(self) -> str:
        from aibo.core.package import Package

        return Package.get_openai_function_name(
            package=self.package,
            function=self.function,
        )

    async def to_openai(self, *, openai_model: OpenAIModel) -> OpenAIContent:
        return {
            "type": "text",
            "text": json.dumps(self.response, indent=2),
        }

    def __str__(self) -> str:
        return json.dumps(self.response, indent=2)


def stringify_message_contents(contents: list[MessageContent]) -> str:
    return "\n\n".join(str(content) for content in contents)


FunctionResponseContent.model_rebuild()

MessageContent = Annotated[
    Union[
        CompletionErrorContent,
        FunctionRequestContent,
        FunctionResponseContent,
        TextMessageContent,
        ImageMessageContent,
    ],
    Field(discriminator="kind"),
]
MessageContentAdapter = TypeAdapter(MessageContent)

MessageContents = list[MessageContent]
MessageContentsAdapter = TypeAdapter(MessageContents)
