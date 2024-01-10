"""
Get lists of what bots are available.
"""
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.pagination import AsyncCursorPage
from openai.types.beta import Assistant

from chats.utils import show_json

load_dotenv()


class InventoryClient:
    def __init__(self):
        self.client = AsyncOpenAI()
        self.model = "gpt-3.5-turbo"

    async def list_assistants(self) -> AsyncCursorPage[Assistant]:
        current_assistants = await self.client.beta.assistants.list()
        return current_assistants

    async def delete_assistant(self, assistant: Assistant) -> None:
        if isinstance(assistant, str):
            raise Exception("uh oh")
        result = await self.client.beta.assistants.delete(assistant_id=assistant.id)
        return result

    async def list_models(self):
        """Doesn't return anything usable?"""
        return self.client.models.list()


async def run():
    client = InventoryClient()
    # models = await client.list_models()
    # print("List models")
    # show_json(models)
    print()
    assistants = await client.list_assistants()
    print("List active assistants")
    show_json(assistants)


if __name__ == "__main__":
    # Python 3.7+
    asyncio.run(run())
