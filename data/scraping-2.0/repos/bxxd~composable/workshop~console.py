import logging
import asyncio
from termcolor import colored

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langwave.memory import VolatileChatMemory
from langchain.chains import ConversationalRetrievalChain

from langchain.chains import LLMChain

log = logging.getLogger(__name__)


async def streaming_console(args):
    chat = ChatOpenAI(
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0,
        verbose=args.debug,
    )
    history = VolatileChatMemory()
    user_input = args.initial

    while True:
        if user_input:
            history.add_user_message(user_input)
            resp = await chat.apredict_messages(history.messages)
            print("\n")
            # log.info(f"AI: {resp} and type is {type(resp)}")
            history.add_message(resp)

        user_input = input(colored(">>>: ", "green"))
        if user_input == "exit":
            break


from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)

from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationChain


async def streaming_chain_console(args):
    chat = ChatOpenAI(
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0,
        verbose=args.debug,
    )
    history = VolatileChatMemory()
    user_input = args.initial

    # human_message_prompt = HumanMessagePromptTemplate(
    #     prompt=PromptTemplate(
    #         template="{input}",
    #         input_variables=["input"],
    #     )
    # )
    # chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])

    # chain = LLMChain(llm=chat, prompt=chat_prompt_template, verbose=True)

    chain = ConversationChain(llm=chat, verbose=True)

    # chain = ConversationalRetrievalChain.from_llm(llm=chat, verbose=True)

    while True:
        if user_input:
            history.add_user_message(user_input)
            # resp = await chat.apredict_messages(history.messages)
            resp = await chain.arun(user_input)
            print("\n")
            log.info(f"AI: {resp} and type is {type(resp)}")
            history.add_message(resp)

        user_input = input(colored(">>>: ", "green"))
        if user_input == "exit":
            break


async def main(args):
    log.info("Hello there!")
    # test_memory()
    await streaming_chain_console(args)


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--initial",
        "-i",
        type=str,
        default="",
        help="Initial message to send to the chatbot",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
