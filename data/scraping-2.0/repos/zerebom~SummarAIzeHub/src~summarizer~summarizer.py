import os

import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


def summarize_text(text, api_key=None):
    # APIキーの設定
    openai.api_key = api_key or os.getenv("OPENAI_API_KEY")

    chat = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", request_timeout=120)
    messages = [HumanMessage(content=text)]
    response = chat(messages)

    return response.content


def dummy_summarize_text(text, api_key=None):
    print(text)
    return "dummy"
