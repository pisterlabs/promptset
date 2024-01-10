from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=apikey,
    model="gpt-4",
    temperature=0
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

prompt = HumanMessage(
    content="I'd like to know more about the city you just mentioned."
)

# add to messages

messages.append(prompt)

response = llm(messages)