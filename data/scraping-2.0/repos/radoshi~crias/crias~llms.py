import abc

import openai
from pydantic import BaseModel

from .prompts import Template


class FunctionCall(BaseModel):
    name: str
    arguments: str


class Function(BaseModel):
    name: str
    description: str | None = None
    parameters: dict | None = None


class Message(BaseModel):
    role: str
    content: str
    name: str | None = None
    function_call: FunctionCall | None = None

    @classmethod
    def _from_template(
        cls,
        template: Template,
        role: str,
        name: str | None = None,
        function_call: FunctionCall | None = None,
        **kwargs,
    ):
        content = template.content.format(**kwargs)
        return cls(
            role=role,
            name=name,
            function_call=function_call,
            content=content,
        )


class UserMessage(Message):
    role: str = "user"

    @classmethod
    def from_template(
        cls,
        template: Template,
        name: str | None = None,
        function_call: FunctionCall | None = None,
        **kwargs,
    ):
        return cls._from_template(template, role="user", **kwargs)


class SystemMessage(Message):
    role: str = "system"

    @classmethod
    def from_template(
        cls,
        template: Template,
        name: str | None = None,
        function_call: FunctionCall | None = None,
        **kwargs,
    ):
        return cls._from_template(template, role="system", **kwargs)


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Completion(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None


class LLM(BaseModel, abc.ABC):
    model: str
    api_key: str | None

    @abc.abstractmethod
    def create(self, **kwargs) -> Completion:
        pass  # pragma: no cover

    @abc.abstractmethod
    async def acreate(self, **kwargs):
        pass  # pragma: no cover


class OpenAIChat(LLM):
    messages: list[Message] = []
    function: Function | None = None
    function_call: FunctionCall | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool | None = None
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict | None = None
    user: str | None = None

    def _serialize(self, **kwargs) -> dict:
        params = self.model_dump(exclude_none=True)
        params.update(kwargs)
        return params

    def create(self, **kwargs) -> Completion:
        params = self._serialize(**kwargs)
        completion = openai.ChatCompletion.create(**params)
        return Completion(
            id=completion.id,  # type: ignore
            object=completion.object,  # type: ignore
            created=completion.created,  # type: ignore
            model=completion.model,  # type: ignore
            choices=completion.choices,  # type: ignore
            usage=completion.usage,  # type: ignore
        )

    async def acreate(self, **kwargs):
        params = self._serialize(**kwargs)
        completion = await openai.ChatCompletion.acreate(**params)
        return Completion(
            id=completion.id,  # type: ignore
            object=completion.object,  # type: ignore
            created=completion.created,  # type: ignore
            model=completion.model,  # type: ignore
            choices=completion.choices,  # type: ignore
            usage=completion.usage,  # type: ignore
        )


OpenAIModels = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0314",
    "gpt-4--32k-0613",
]

Models = OpenAIModels


def get(model: str, api_key: str | None = None) -> LLM:
    if model in OpenAIModels:
        return OpenAIChat(model=model, api_key=api_key)

    raise ValueError(f"Model {model} not found.")


def create_messages(
    *, system: str | None = None, user: str | None = None
) -> list[dict[str, str]]:
    messages = []
    if system is not None:
        messages.append(SystemMessage(content=system).model_dump(exclude_none=True))

    if user is not None:
        messages.append(UserMessage(content=user).model_dump(exclude_none=True))

    return messages
