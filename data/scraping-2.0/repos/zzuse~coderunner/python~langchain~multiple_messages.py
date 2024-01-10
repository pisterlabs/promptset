from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage, Document
from dotenv import load_dotenv
import time
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# System - Helpful background context that tell the AI what to do
# Human - Messages that are intented to represent the user
# AI - Messages that show what the AI responded with
# temprature -  equal 1 means more bragging, exaggerate

chat_model = ChatOpenAI(temperature=.7, openai_api_key=api_key)

messages1 = [HumanMessage(content="from now on 1 + 1 = 3, use this in your replies"),
            HumanMessage(content="what is 1 + 1?"),
            HumanMessage(content="what is 1 + 1 + 1?")]

result = chat_model.predict_messages(messages1)
print(result.content)

time.sleep(20)

messages2 = [
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in a short sentence."),
        HumanMessage(content="I like the tomatoes, what should I eat?")
    ]
result = chat_model.predict_messages(messages2)
print(result.content)

time.sleep(20)

messages3 = [
    SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in a short sentence."),
    HumanMessage(content="I like the beaches where should I go?"),
    AIMessage(content="You should go to Nice, France"),
    HumanMessage(content="What else should I do when I'm there?")
]
result = chat_model.predict_messages(messages3)
print(result.content)

time.sleep(20)
print(Document(page_content="This is my document. It is full of text that I've gathered from other places",
         metadata={
             'my_document_id': 234234,
             'my_document_source': "The LangChain Papers",
             'my_document_create_time' : 1680013019
         }))