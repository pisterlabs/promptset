from typing import Literal, Optional

from dynamodm import DynaModel  # type: ignore
from openai.types.beta.threads import ThreadMessage
from pydantic import BaseModel, Field  # pylint: disable=E0611

from ._service import Service


class ThreadMessageRequest(BaseModel):
    thread_id: str = Field(
        default=...,
        description="The thread ID of the thread to use for the thread message",
    )
    content: str = Field(default=..., description="The content of the thread message")
    role: Optional[Literal["user"]] = Field(
        default="user", description="The role of the thread message"
    )
    file_ids: Optional[list[str]] = Field(
        default=None, description="The file IDs of the thread message"
    )


class ThreadMessageModel(DynaModel):
    key: str = Field(..., pk=True)
    sort: str = Field(..., sk=True)
    data: ThreadMessage


class ThreadMessageService(Service[ThreadMessageRequest, ThreadMessage]):
    @property
    def client(self):
        return self.api.beta.threads.messages

    async def create_(self, *, key: str, data: ThreadMessageRequest):
        """
        Creates a thread message.
        """
        response = await self.client.create(**data.dict())
        instance = await ThreadMessageModel(
            key=key, sort=response.id, data=response
        ).put()
        return instance.data

    async def list_(self, *, key: str):
        """
        Lists all thread messages.
        """
        response = await ThreadMessageModel.query(pk=key)
        return [item.data for item in response]

    async def get_(self, *, key: str, sort: str):
        """
        Gets a thread message.
        """
        response = await ThreadMessageModel.get(pk=key, sk=sort)
        return response.data

    async def update_(self, *, key: str, sort: str, data: ThreadMessageRequest):
        """
        Updates a thread message.
        """
        response = await self.client.update(**data.dict())
        instance = await ThreadMessageModel(key=key, sort=sort, data=response).put()
        return instance.data

    async def delete_(self, *, key: str, sort: str):
        """
        Deletes a thread message.
        """
        await ThreadMessageModel.delete(pk=key, sk=sort)
