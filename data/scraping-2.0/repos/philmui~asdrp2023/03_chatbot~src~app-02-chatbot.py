import asyncio
import chainlit as cl
import os
from langchain.chat_models import (
    ChatOpenAI, 
    ChatGooglePalm, 
    ChatAnthropic
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
MODEL_NAME = "gpt-3.5-turbo"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

system_prompt="""
You are a helpful assistant that truthfully respond to a user's query or question.

If you don't know the answer, simply answer: I don't know. 
Most importantly, do not respond with false information.
"""

@cl.on_message # for every user message
def main(query: str):
    messages = [
        {'role':'system', 'content':system_prompt},
        {'role':'user', 'content':query}
    ]
    response_text=""
    try:
        chat = ChatOpenAI(temperature=0, model=MODEL_NAME)
        response = chat.predict_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ]
        )
        response_text=response.content
    except Exception as e:
        response_text=f"no response: {e}"

    # final answer
    asyncio.run(
        cl.Message(
            content=response_text
        ).send()
    )
    
@cl.on_chat_start
def start():
    asyncio.run(
        cl.Message(
            content="Hello there!"
        ).send()
    ) 