import asyncio
from typing import Any, Dict, List

from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import LLMResult


class MyCustomSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")


class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from chatbot."""

    async def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_new_token(self, token: str, **kwargs) -> str:
        return f"async handler: token: {token}"

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is ending")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    # To enable streaming, we pass in `streaming=True` to the ChatModel constructor
    # Additionally, we pass in a list with our custom handler
    llm = ChatOpenAI(
        max_tokens=25,
        streaming=True,
        callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()],
    )


    async def stream_generator():
        async for token in llm.astream("Say 3 words"):
            print(f"wowooo - {token.content}")


    asyncio.run(stream_generator())
