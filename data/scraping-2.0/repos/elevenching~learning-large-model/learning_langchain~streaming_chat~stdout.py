import asyncio
import sys

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage

from learning_langchain.multi_key_chat_open_ai import MultiKeyChatOpenAI

handler = AsyncIteratorCallbackHandler()
llm = MultiKeyChatOpenAI(streaming=True, callbacks=[handler], temperature=0, verbose=True)


async def consumer():
    iterator = handler.aiter()
    async for item in iterator:
        sys.stdout.write(item)
        sys.stdout.flush()


async def producer():
    while True:
        message = input("请输入您的问题：")
        await llm.agenerate(messages=[[HumanMessage(content=message)]])
        sys.stdout.write("\n")
        sys.stdout.flush()


async def main():
    consumer_task = asyncio.create_task(consumer())
    producer_task = asyncio.create_task(producer())
    await asyncio.gather(consumer_task, producer_task)


if __name__ == '__main__':
    asyncio.run(main())
