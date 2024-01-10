from operator import itemgetter
import chainlit as cl
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from retriever import retriever
from prompts import answer_prompt

@cl.on_chat_start
async def on_chat_start():
    USE_GPT4 = False  # set to true if relevant
    model_name = "gpt-4-1106-preview" if USE_GPT4 else "gpt-3.5-turbo-1106"
    retriever_instance = retriever  # rename to avoid conflict with module name
    prompt = answer_prompt
    model = ChatOpenAI(
        model=model_name, # this can be changed to any model available at OpenAI
        temperature=0.2,  # precise mode
        callbacks=[StreamingStdOutCallbackHandler()],  # allows for streaming if LLM supports it
        streaming=True  # allows for streaming if LLM supports it
    )
    global runnable  # declare as global to be accessible in on_message
    runnable = (
        {
            "context": itemgetter("question") | retriever_instance,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()