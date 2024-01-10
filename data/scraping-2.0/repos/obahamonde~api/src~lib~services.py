from abc import ABC, abstractmethod
from typing import Any, List, Optional

from openai import AsyncOpenAI
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, Field  # pylint: disable=E0611

from .decorators import handle
from .schemas import IMessage
from .vector import VectorClient


class OpenAIFunction(BaseModel, ABC):
    """
    Base class for OpenAI functions.

    Attributes:
            definition (FunctionDefinition): The definition of the function.
    """

    @classmethod
    def definition(cls) -> FunctionDefinition:
        """Returns a FunctionDefinition object for the current service method.

        Raises:
                AssertionError: If the method's docstring is None.

        Returns:
                FunctionDefinition: A FunctionDefinition object containing the method's name, description, and parameters.
        """
        assert cls.__doc__ is not None, "OpenAIFunction must have a docstring"
        _schema = cls.schema()  # type: ignore
        _name = cls.__name__
        _description = cls.__doc__
        _parameters = {
            "type": "object",
            "properties": {
                k: v for k, v in _schema["properties"].items() if k != "self"
            },
            "required": _schema.get("required", []),
        }
        return FunctionDefinition(name=_name, description=_description, parameters=_parameters)  # type: ignore

    @abstractmethod
    async def run(self) -> Any:
        raise NotImplementedError

    @handle
    async def __call__(self):
        return await self.run()


class Stack(BaseModel):
    tools: Optional[List[FunctionDefinition]] = Field(default=None)
    messages: Optional[List[IMessage]] = Field(default=None)

    @property
    def ai(self) -> AsyncOpenAI:
        """
        Returns an instance of AsyncOpenAI.
        """
        return AsyncOpenAI()

    @property
    def db(self) -> VectorClient:
        """
        Returns an instance of the VectorClient class.
        """
        return VectorClient()
