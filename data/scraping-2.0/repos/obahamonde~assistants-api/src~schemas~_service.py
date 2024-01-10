from abc import ABC, abstractmethod
from typing import AsyncIterable, Generic, Iterable, TypeVar, Union

from glob_utils import robust  # type: ignore
from openai import AsyncOpenAI
from pydantic import BaseModel  # pylint: disable=E0611

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)
Page = Union[Iterable[I], AsyncIterable[I]]


class Service(Generic[I, O], ABC):
    @property
    def api(self) -> AsyncOpenAI:
        return AsyncOpenAI()

    @abstractmethod
    @robust
    async def create_(self, *, key: str, data: I) -> O:
        ...

    @abstractmethod
    @robust
    async def list_(self, *, key: str) -> Page[O]:
        ...

    @abstractmethod
    @robust
    async def get_(self, *, key: str, sort: str) -> O:
        ...

    @abstractmethod
    @robust
    async def update_(self, *, key: str, sort: str, data: I) -> O:
        ...

    @abstractmethod
    @robust
    async def delete_(self, *, key: str, sort: str) -> None:
        ...
