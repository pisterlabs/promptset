import chainlit as cl
from embedchain import App
from embedchain.store.assistants import OpenAIAssistant
import os


from dotenv import load_dotenv

load_dotenv()

import os

@cl.on_chat_start
async def on_chat_start():
    assistant = OpenAIAssistant(assistant_id="asst_LsEbMB7nz2squzGOtndlmezI")
    # import your data here
    assistant.collect_metrics = False
    cl.user_session.set("app", assistant)


@cl.on_message
async def on_message(message: cl.Message):
    app = cl.user_session.get("app")
    msg = cl.Message(content="")
    for chunk in await cl.make_async(app.chat)(message.content):
        await msg.stream_token(chunk)

    await msg.send()
