import chainlit as cl
from langchain import PromptTemplate, LLMChain, OpenAI
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate

from langchain.document_loaders import ArxivLoader

template = """Please create a social media post on LinkedIn with emojis for the academic paper described below.

Paper description: {paper_description}
"""


@cl.on_chat_start
async def start():
    content = "🙋Hi there! I will assist you 💁 in creating a social media post for an academic paper 📝 \n\n Paste paper information."
    await cl.Message(content=content).send()


@cl.langchain_factory(use_async=False)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["paper_description"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0, streaming=True), verbose=True)

    return llm_chain
