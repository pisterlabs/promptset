import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

_ = load_dotenv(find_dotenv())  # read local .env file


# ---------------------------
# Chat
# OPEN AI API - POST https://api.openai.com/v1/chat/completions
# ---------------------------

chat = ChatOpenAI(temperature=0)

batch_messages = [
    [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="I love programming."),
    ],
    [
        SystemMessage(
            content="You are a helpful assistant that translates English to French."
        ),
        HumanMessage(content="I love artificial intelligence."),
    ],
]
result = chat.generate(batch_messages)
print(result)
print(result.llm_output["token_usage"])
