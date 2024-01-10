import chainlit as cl
import openai
import os
from langchain import PromptTemplate, OpenAI, LLMChain

os.environ["OPENAI_API_KEY"] = "ADD_YOUR_API_KEY_HERE"

template = """Question: {question}

Answer: Let's think step by step."""

@cl.on_chat_start
def main():
    prompt = PromptTemplate(template = template, input_variables = ["question"])
    llmchain = LLMChain(
        prompt = prompt,
        llm = OpenAI(temperature = 0.75 , streaming = True),
        verbose = True
    )
    cl.user_session.set("llmchain", llmchain)

@cl.on_message
async def main(message : str):
    llmchain = cl.user_session.get("llmchain")
    response = await llmchain.acall(message, callbacks = [cl.AsyncLangchainCallbackHandler()])
    await cl.Message(content = response["text"]).send()