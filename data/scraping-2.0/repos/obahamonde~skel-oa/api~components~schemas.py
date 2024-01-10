from __future__ import annotations

import os
from abc import ABC, abstractmethod
from base64 import b64encode
from typing import Any, Generic, Literal, Optional, TypeVar

from openai import AsyncOpenAI
from openai.types.beta.assistant import Assistant
from openai.types.beta.assistant_create_params import (
    ToolAssistantToolsCode,
    ToolAssistantToolsFunction,
    ToolAssistantToolsRetrieval,
)
from openai.types.beta.thread import Thread
from openai.types.beta.thread_create_params import Message
from openai.types.beta.threads import ThreadMessage
from openai.types.beta.threads.run import Run
from openai.types.file_object import FileObject
from pydantic import BaseConfig, BaseModel, Field

from .decorators import robust

T = TypeVar("T", bound=BaseModel)

basic_tools = [
    ToolAssistantToolsCode(type="code_interpreter"),
    ToolAssistantToolsRetrieval(type="retrieval"),
]


class BaseResource(BaseModel, ABC, Generic[T]):
    metadata: Optional[dict[str, Any]] = Field(default=None)

    @classmethod
    def api(cls):
        return AsyncOpenAI()

    class Config(BaseConfig):
        orm_mode = True

    @abstractmethod
    @robust
    async def post(self) -> T:
        ...

    def dict(self, *args: Any, **kwargs: Any):
        if self.__class__ is FileResource:
            self.metadata = None
        return super().dict(*args, exclude_none=True, **kwargs)


class FileResource(BaseResource[FileObject]):
    file: bytes = Field(...)
    purpose: Literal["assistants"] = Field(default="assistants")

    async def post(self) -> FileObject:
        return await self.api().files.create(**self.dict())

    @classmethod
    @robust
    async def get(cls, *, file_id: str) -> FileObject:
        return await cls.api().files.retrieve(file_id=file_id)

    @classmethod
    @robust
    async def delete(cls, *, file_id: str):
        await cls.api().files.delete(file_id=file_id)


class AssistantResource(BaseResource[Assistant]):
    model: str = Field(...)
    description: Optional[str] = Field(default=None)
    file_ids: Optional[list[str]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    instructions: Optional[str] = Field(default=None)
    tools: Optional[list[ToolAssistantToolsFunction]] = Field(default=None)

    async def post(self) -> Assistant:
        if self.tools is None:
            self.tools = basic_tools  # type: ignore
        else:
            self.tools = basic_tools + self.tools  # type: ignore
        return await self.api().beta.assistants.create(
            **self.dict(),
        )

    @classmethod
    @robust
    async def get(cls, *, assistant_id: str) -> Assistant:
        return await cls.api().beta.assistants.retrieve(assistant_id=assistant_id)

    @classmethod
    @robust
    async def delete(cls, *, assistant_id: str):
        await cls.api().beta.assistants.delete(assistant_id=assistant_id)

    async def put(self, *, assistant_id: str):
        return await self.api().beta.assistants.update(
            assistant_id=assistant_id, **self.dict()
        )


class ThreadResource(BaseResource[Thread]):
    messages: Optional[list[Message]] = Field(default=None)

    async def post(self) -> Thread:
        return await self.api().beta.threads.create(**self.dict())

    @classmethod
    @robust
    async def get(cls, *, thread_id: str) -> Thread:
        return await cls.api().beta.threads.retrieve(thread_id=thread_id)

    @classmethod
    @robust
    async def delete(cls, *, thread_id: str):
        await cls.api().beta.threads.delete(thread_id=thread_id)


class ThreadMessageResource(BaseResource[ThreadMessage]):
    thread_id: str = Field(...)
    content: str = Field(...)
    role: Literal["user"] = Field(default="user")
    file_ids: Optional[list[str]] = Field(default=None)

    async def post(self) -> ThreadMessage:
        return await self.api().beta.threads.messages.create(**self.dict())

    @classmethod
    @robust
    async def get(cls, *, message_id: str, thread_id: str) -> ThreadMessage:
        return await cls.api().beta.threads.messages.retrieve(
            message_id=message_id, thread_id=thread_id
        )

    @classmethod
    @robust
    async def list(cls, *, thread_id: str):
        return await cls.api().beta.threads.messages.list(thread_id=thread_id)


class RunResource(BaseResource[Run]):
    thread_id: str = Field(...)
    assistant_id: str = Field(...)
    instructions: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    tools: Optional[list[ToolAssistantToolsFunction]] = Field(default=None)

    async def post(self) -> Run:
        if self.tools is None:
            self.tools = basic_tools  # type: ignore
        else:
            self.tools = basic_tools + self.tools  # type: ignore
        return await self.api().beta.threads.runs.create(
            **self.dict(),
        )

    @classmethod
    @robust
    async def get(cls, *, run_id: str, thread_id: str) -> Run:
        return await cls.api().beta.threads.runs.retrieve(
            run_id=run_id, thread_id=thread_id
        )

    @classmethod
    @robust
    async def delete(cls, *, run_id: str, thread_id: str):
        return await cls.api().beta.threads.runs.cancel(
            run_id=run_id, thread_id=thread_id
        )

    @classmethod
    @robust
    async def list(cls, *, thread_id: str, run_id: str):
        return await cls.api().beta.threads.runs.steps.list(
            thread_id=thread_id, run_id=run_id
        )


class Node(BaseModel):
    path: str
    name: str
    type: str
    content: list[Node] | str

    @classmethod
    def tree(cls, path: str = ".") -> Node:
        name = os.path.basename(path)
        if os.path.isdir(path):
            return cls(
                path=path,
                name=name,
                type="directory",
                content=[cls.tree(os.path.join(path, x)) for x in os.listdir(path)],
            )
        else:
            with open(path, "rb") as f:
                try:
                    content = f.read().decode()
                except UnicodeDecodeError:
                    content = b64encode(f.read()).decode()
            return cls(
                path=path,
                name=name,
                type="file",
                content=content,
            )
