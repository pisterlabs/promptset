from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl
from app import init_prompt, template


@cl.on_chat_start
async def init():
    prompt = PromptTemplate(template=init_prompt + template, input_variables=["note"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

    cl.user_session.set("llm_chain", llm_chain)
    await cl.Message(content="Start by adding a 'note'").send()


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["text"]).send()
