from dynamodm import DynaModel  # type: ignore
from openai.types.beta.thread import Thread
from openai.types.beta.thread_create_params import Message
from pydantic import BaseModel, Field  # pylint: disable=E0611

from ._service import Service


class ThreadRequest(BaseModel):
    messages: list[Message] = Field(
        default=..., description="The messages to use for the thread"
    )


class ThreadResponse(
    Thread,
):
    title: str = Field(default=..., description="The title of the thread")


class ThreadModel(DynaModel):
    key: str = Field(..., pk=True)
    sort: str = Field(..., sk=True)
    data: ThreadResponse


class ThreadService(Service[ThreadRequest, ThreadResponse]):
    @property
    def client(self):
        return self.api.beta.threads

    async def create_(self, *, key: str, data: ThreadRequest) -> ThreadResponse:
        try:
            response = await self.client.create(**data.dict())
            text = data.messages[0]["content"]
            output = await ThreadModel(
                key=key,
                sort=response.id,
                data=ThreadResponse(
                    **response.dict(), title=await self._get_title(text=text)
                ),
            ).put()
            return output.data
        except IndexError:
            response = await self.client.create()
            return (
                await ThreadModel(
                    key=key,
                    sort=response.id,
                    data=ThreadResponse(**response.dict(), title="[New Thread]"),
                ).put()
            ).data

    async def list_(self, *, key: str) -> list[ThreadResponse]:
        array = await ThreadModel.query(pk=key)
        return [item.data for item in array]

    async def get_(self, *, key: str, sort: str) -> ThreadResponse:
        res = await ThreadModel.get(pk=key, sk=sort)
        return res.data

    async def update_(
        self, *, key: str, sort: str, data: ThreadRequest
    ) -> ThreadResponse:
        return await self.create_(key=key, data=data)

    async def delete_(self, *, key: str, sort: str):
        await self.client.delete(thread_id=sort)
        await ThreadModel.delete(pk=key, sk=sort)

    async def _get_title(self, *, text: str):
        """
        Get the title of the thread.

        Args:
                        text: The text to use for the thread.

        Returns:
                        The title of the thread.
        """
        return (
            (
                await self.api.completions.create(
                    prompt=f"You are conversations title generator. The conversation first message is: '{text}' The title of the thread is:",
                    model="gpt-3.5-turbo-instruct",
                )
            )
            .choices[0]
            .text
        )
