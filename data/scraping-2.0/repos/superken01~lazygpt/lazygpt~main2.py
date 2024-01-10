import os

import chainlit as cl
from agent import create_agent
from langchain.tools import ShellTool
from langchain.tools.file_management import FileSearchTool, ReadFileTool, WriteFileTool
from langchain_experimental.tools import PythonREPLTool
from tools.google import GoogleSearchTool
from tools.web import BrowseWebSiteTool

os.makedirs(".lazygpt", exist_ok=True)


@cl.on_chat_start
def start_chat():
    pass


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    director = cl.user_session.get("director")
    if agent is None:
        tools = [
            GoogleSearchTool(),
            BrowseWebSiteTool(),
            PythonREPLTool(),
        ]
        agent = create_agent(
            # system_message="你只會說中文 你將依照我的提示一步一步找出問題的答案 找出答案後回覆`TERMINATE`來結束對話",
            system_message="你將依照我的指示一步一步找出問題的答案 若指示無法執行或不合理 請回覆理由並且要求重新指示 請不要給我提示",
            tools=tools,
            streaming=True,
            verbose=True,
        )

        tools = [
            PythonREPLTool(),
        ]
        director = create_agent(
            system_message="你的任務是一步一步指導我讓我自己找出問題的答案 我給你的資訊可能不完整或不正確請指正我 我只會使用google搜尋 也可以瀏覽網頁內容 一次只給一個提示",
            tools=tools,
            streaming=True,
            verbose=True,
        )
        cl.user_session.set("agent", agent)
        cl.user_session.set("director", director)

    agent.memory.chat_memory.add_user_message(message.content)

    callback = cl.LangchainCallbackHandler(stream_final_answer=True)
    result = await cl.make_async(director)(
        {"input": message.content}, callbacks=[callback]
    )
    output = result["output"]
    await cl.Message(content=output, author="Director").send()

    while True:
        callback = cl.LangchainCallbackHandler(stream_final_answer=True)
        result = await cl.make_async(agent)({"input": output}, callbacks=[callback])
        output = result["output"]
        await cl.Message(content=output, author="Agent").send()

        callback = cl.LangchainCallbackHandler(stream_final_answer=True)
        result = await cl.make_async(director)({"input": output}, callbacks=[callback])
        output = result["output"]
        await cl.Message(content=output, author="Director").send()
