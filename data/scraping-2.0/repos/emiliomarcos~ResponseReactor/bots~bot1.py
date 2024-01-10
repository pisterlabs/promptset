import os
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

def run():
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0.3)

    messages = [
        SystemMessage(content="You are an expert data scientist"),
        HumanMessage(content="Write a Python script that trains a neural network on simulated data")
    ]

    response = chat(messages)

    print(response.content, end='\n')
    return response.content
