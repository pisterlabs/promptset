import asyncio
import os
import chainlit as cl
from langchain.prompts import (
    PromptTemplate,
)
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
MODEL_NAME = "text-davinci-003"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

prompt_template="""
You are a helpful assistant that truthfully respond to a user's query or question.

User's query: {query}

If you don't know the answer, simply answer: I don't know. 
Most importantly, do not respond with false information.
"""

prompt=PromptTemplate(
    input_variables=["query"],
    template=prompt_template
)

@cl.on_message # for every user message
def main(message: str):
    llm = OpenAI(openai_api_key=OPENAI_API_KEY,
                 model=MODEL_NAME)
    response = llm(prompt.format(query=message))

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