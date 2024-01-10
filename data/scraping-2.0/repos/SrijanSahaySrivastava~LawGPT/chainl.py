from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

from model import final_result
from model import qa_bot


# Chain lit
@cl.on_chat_start
async def on_chat_start():
    files = None
    chain = qa_bot()
    msg = cl.Message(content="Starting Bharatgpt!")
    await msg.send()
    msg.content = "Hi, welcome to the BHARATGPT. what is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["Answer:"],
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\n\nSource document: " + str(sources)
    else:
        answer += f"\nNo source document found"

    await cl.Message(content=answer).send()
