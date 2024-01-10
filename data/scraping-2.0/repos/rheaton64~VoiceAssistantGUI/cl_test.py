import os
from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl

template = """Question: {question}

Answer: Let's think step by step."""


@cl.on_message
def on_message(message):
    print(message)
    cl.send_message(content='Hello', author='Human')