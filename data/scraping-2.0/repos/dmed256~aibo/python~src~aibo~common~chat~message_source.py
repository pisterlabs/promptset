from typing import Annotated, Literal, Optional, Self, Union

from pydantic import BaseModel, Field, TypeAdapter

from aibo.common.constants import Env
from aibo.common.openai import OPENAI_MODELS_BY_MODEL, OpenAIModel

__all__ = [
    "HumanSource",
    "MessageSource",
    "OpenAIModelSource",
    "ProgrammaticSource",
    "MessageSourceAdapter",
]


class HumanSource(BaseModel):
    """
    The source of the message came from a human
    """

    kind: Literal["human"] = "human"
    user: str

    def __str__(self) -> str:
        return f"human:{self.user}"


class OpenAIModelSource(BaseModel):
    """
    The source of the message came from an OpenAI model
    """

    kind: Literal["openai_model"] = "openai_model"
    model: str
    temperature: float
    max_tokens: Optional[int]

    @classmethod
    def build(
        cls,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Self:
        env = Env.get()
        return cls(
            model=model or env.OPENAI_MODEL,
            temperature=env.OPENAI_TEMPERATURE if temperature is None else temperature,
            max_tokens=max_tokens,
        )

    @property
    def openai_model(self) -> OpenAIModel:
        return OPENAI_MODELS_BY_MODEL[self.model]

    def __str__(self) -> str:
        return f"model:{self.model}"


class ProgrammaticSource(BaseModel):
    """
    There was no explicit source of the message (e.g. function-generated)
    """

    kind: Literal["programmatic"] = "programmatic"
    source: str

    def __str__(self) -> str:
        return f"programmatic:{self.source}"


MessageSource = Annotated[
    Union[
        HumanSource,
        OpenAIModelSource,
        ProgrammaticSource,
    ],
    Field(discriminator="kind"),
]
MessageSourceAdapter = TypeAdapter(MessageSource)
