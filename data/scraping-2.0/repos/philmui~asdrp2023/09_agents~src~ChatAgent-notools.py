import chainlit as cl
import os
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
MODEL_NAME = "text-davinci-003"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

def chat(query: str):
    llm = OpenAI(openai_api_key=OPENAI_API_KEY,
                 model=MODEL_NAME,
                 temperature=0)
    return llm(query)
    
@cl.on_message # for every user message
async def main(query: str):
    # final answer
    await cl.Message(
            content=chat(query)
        ).send()
    
@cl.on_chat_start
async def start():
    await cl.Message(
            content="Hello there!"
        ).send()