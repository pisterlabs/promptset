from typing import TYPE_CHECKING, Optional
from openai.types.beta.threads import ThreadMessage
from pydantic import BaseModel, Field

from cyberchipped.utilities.asyncio import (
    ExposeSyncMethodsMixin,
    expose_sync_method,
)
from cyberchipped.utilities.logging import get_logger
from cyberchipped.utilities.openai import get_client
from cyberchipped.utilities.pydantic import parse_as

logger = get_logger("Threads")

if TYPE_CHECKING:
    from .assistants import Assistant
    from .runs import Run


class Thread(BaseModel, ExposeSyncMethodsMixin):
    id: Optional[str] = None
    metadata: dict = {}
    messages: list[ThreadMessage] = Field([], repr=False)

    @expose_sync_method("get")
    async def get_async(self):
        """
        Gets a thread.
        """
        client = get_client()
        response = await client.beta.threads.retrieve(thread_id=self.id)
        self.id = response.id
        return self

    @expose_sync_method("create")
    async def create_async(self=None):
        """
        Creates a thread.
        """
        client = get_client()
        response = await client.beta.threads.create()
        self.id = response.id
        return self

    @expose_sync_method("add")
    async def add_async(
        self, message: str, file_paths: Optional[list[str]] = None
    ) -> ThreadMessage:
        """
        Add a user message to the thread.
        """
        client = get_client()

        # Create the message with the attached files
        response = await client.beta.threads.messages.create(
            thread_id=self.id,
            role="user",
            content=message,
        )
        return ThreadMessage.model_validate(response.model_dump())

    @expose_sync_method("get_messages")
    async def get_messages_async(
        self,
        limit: int = None,
        before_message: Optional[str] = None,
        after_message: Optional[str] = None,
    ):
        client = get_client()

        response = await client.beta.threads.messages.list(
            thread_id=self.id,
            # note that because messages are returned in descending order,
            # we reverse "before" and "after" to the API
            before=after_message,
            after=before_message,
            limit=limit,
            order="desc",
        )

        return parse_as(list[ThreadMessage], reversed(response.model_dump()["data"]))

    @expose_sync_method("delete")
    async def delete_async(self):
        client = get_client()
        await client.beta.threads.delete(thread_id=self.id)
        self.id = None

    @expose_sync_method("cancel_run")
    async def cancel_run_async(
        self,
        assistant: "Assistant",
        **run_kwargs,
    ) -> "Run":
        """
        Cancels the run of this thread with the provided assistant.
        """
        from cyberchipped.assistants.runs import Run

        run = Run(assistant=assistant, thread=self, **run_kwargs)
        return await run.cancel_async()

    @expose_sync_method("say")
    async def say_async(self, text: str, assistant: "Assistant") -> str:
        """
        Wraps the full process of adding a message to the thread and running it
        """
        from cyberchipped.assistants.runs import Run

        try:
            await self.cancel_run_async(assistant=assistant)
        except Exception:
            pass

        await self.add_async(text)

        run = Run(assistant=assistant, thread=self)
        await run.run_async()
        messages = await self.get_messages_async()
        last_message = messages[-1]
        ai_message = last_message.content[0].text.value
        return ai_message
