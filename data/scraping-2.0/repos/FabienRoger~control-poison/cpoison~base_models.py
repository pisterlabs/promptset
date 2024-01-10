from attrs import define
from abc import ABC, abstractmethod
from lowstakes.utils import AndMetadata
from lowstakes.llm import OpenAIChatModel, call_llm, Role
from cpoison.config import TRUSTED_MODEL, UNTRUSTED_MODEL
from typing import Optional


class Model(ABC):
    @abstractmethod
    async def call(self, instruction: str, input: str) -> AndMetadata[str]:
        ...

    async def __call__(self, instruction: str, input: str) -> str:
        return (await self.call(instruction, input))[0]


@define
class DirectModel(Model):
    llm: OpenAIChatModel

    async def call(self, instruction: str, input: str) -> AndMetadata[str]:
        r = await call_llm(self.llm, [("system", instruction), ("user", input)])
        return r[0].completion if r else "", {}

    async def calls(self, instruction: str, input: str, n: int) -> list[str]:
        r = await call_llm(self.llm, [("system", instruction), ("user", input)], temperature=1, n=n)
        return [x.completion for x in r]

    @classmethod
    def trusted(cls):
        return cls(llm=TRUSTED_MODEL)

    @classmethod
    def untrusted(cls):
        return cls(llm=UNTRUSTED_MODEL)
