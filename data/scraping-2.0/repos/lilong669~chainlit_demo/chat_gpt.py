import chainlit as cl
from chainlit.prompt import Prompt, PromptMessage
from chainlit.playground.providers.openai import ChatOpenAI

import openai
import os

from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()
template = """你是一个超级AI助手，能提供卓越的AI建议： 请问回答以下问题：{input}: Let’s think step by step
```"""


settings = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["```"],
}

@cl.on_message
async def main(message: str):
    # Create the prompt object for the Prompt Playground
    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="user",
                template=template,
                formatted=template.format(input=message)
            )
        ],
        settings=settings,
        inputs={"input": message},
    )

    # Prepare the message for streaming
    msg = cl.Message(
        content="",
        language="sql",
    )

    # Call OpenAI
    async for stream_resp in await openai.ChatCompletion.acreate(
        messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    ):
        token = stream_resp.choices[0]["delta"].get("content", "")
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()