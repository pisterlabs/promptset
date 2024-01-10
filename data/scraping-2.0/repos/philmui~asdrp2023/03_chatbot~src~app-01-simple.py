import asyncio
import chainlit as cl
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
MODEL_NAME = "text-davinci-003"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

@cl.on_message # for every user message
def main(message: str):
    llm = OpenAI(openai_api_key=OPENAI_API_KEY,
                 model=MODEL_NAME)
    response = llm(message)

    # final answer
    asyncio.run(
        cl.Message(
            content=response
        ).send()
    )
    
@cl.on_chat_start
def start():
    asyncio.run(
        cl.Message(
            content="Hello there!"
        ).send()
    ) 