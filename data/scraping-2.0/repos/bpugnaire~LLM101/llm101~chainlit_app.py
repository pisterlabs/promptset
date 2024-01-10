import chainlit as cl
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from dialog_chain import ask_gigi

load_dotenv()

OPENAI_MODEL = 'gpt-3.5-turbo'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature=0, streaming=True)
prompt_template = PromptTemplate.from_template(
    "Answer the question: {question}"
)

def ask_llm(question):
    prompt = prompt_template.format(question=question)
    response = llm.predict(prompt)
    return response

# @cl.on_message
# async def main(message: cl.Message):
#     # Your custom logic goes here...

#     # Send a response back to the user
#     await cl.Message(
#         content=f"Received: {message.content}",
#     ).send()

@cl.on_message
async def main(message: cl.Message):

    response = ask_gigi(message.content)

    await cl.Message(
        content=response,
        author="Gigi",
    ).send()