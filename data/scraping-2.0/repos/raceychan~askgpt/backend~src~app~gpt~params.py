import typing as ty

from openai._types import Body, Headers, NotGiven, Query
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)
from src.domain.model.base import ValueObject

# TODO: read
# https://learn.microsoft.com/en-us/dotnet/architecture/microservices/microservice-ddd-cqrs-patterns/domain-events-design-implementation


# class ModelEndpoint(BaseModel):
#     endpoint: Path
#     models: tuple[str, ...]


# class CompletionEndPoint(ModelEndpoint):
#     endpoint: Path = Path("/v1/chat/completion")
#     model: CompletionModels

CompletionModels = ty.Literal[
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-16k-0613",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
]

# TODO: use enum
ChatGPTRoles = ty.Literal["system", "user", "assistant", "function"]


class CompletionOptions(ty.TypedDict, total=False):
    messages: ty.List[ChatCompletionMessageParam]
    model: CompletionModels
    frequency_penalty: float | None | NotGiven
    function_call: completion_create_params.FunctionCall | NotGiven
    functions: list[completion_create_params.Function] | NotGiven
    logit_bias: dict[str, int] | None | NotGiven
    max_tokens: int | None | NotGiven
    n: int | None | NotGiven
    presence_penalty: float | None | NotGiven
    response_format: completion_create_params.ResponseFormat | NotGiven
    seed: int | None | NotGiven
    stop: (str | None) | (list[str]) | NotGiven
    stream: bool
    temperature: float | None | NotGiven
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven
    tools: list[ChatCompletionToolParam] | NotGiven
    top_p: float | None | NotGiven
    user: str | NotGiven
    extra_headers: Headers | None
    extra_query: Query | None
    extra_body: Body | None
    timeout: float | None | NotGiven


class ChatResponse(ValueObject):
    """
    a domain representation of chat response
    """

    chunk: ChatCompletionChunk

    async def __aiter__(self) -> ty.Any:
        ...
