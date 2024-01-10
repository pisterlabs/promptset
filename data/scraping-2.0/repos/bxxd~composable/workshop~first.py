import logging
from langwave.memory.volatile import VolatileMemory
import asyncio

from langchain.memory import ChatMessageHistory, ConversationBufferMemory

from langchain.chat_models import ChatOpenAI

from langchain.llms import OpenAI
from langchain.chains import ConversationChain

from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

log = logging.getLogger(__name__)


def test_agent():
    chat = ChatOpenAI()

    resp = chat.predict("hi there")

    log.info(f"chat({chat.model_name}): {resp}")


async def test_conversation():
    chat = ChatOpenAI(
        streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0
    )

    history = ChatMessageHistory()

    history.add_user_message("hi")

    resp = await chat.apredict_messages(history.messages)

    log.info(f"resp: {resp}")

    log.info(f"history messages: {history.messages}")

    llm = OpenAI(temperature=0)
    # conversation = ConversationChain(llm=llm, verbose=True, memory=ChatMessageHistory())

    # resp = conversation.predict(input="Hi there!")

    # log.info(f'conversation: {resp}')
    # log.info(f'memory: {conversation.memory}')


def test_memory():
    memory = VolatileMemory()
    log.info(f"memory: {memory}")


async def main(args):
    log.info("Hello there!")
    # test_memory()
    await test_conversation()


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
