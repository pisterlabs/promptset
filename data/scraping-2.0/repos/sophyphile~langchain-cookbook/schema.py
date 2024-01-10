import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Text
# You'll be working with single strings (that will soon grow in complexity!)
my_text = "What day comes after Friday?"

# Chat
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
# AI may not have actually responded with AIMessage, but you can tell it that it did so it has additional context on how to respond to you.

chat = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key)

# print(chat(
#     [
#         SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
#         HumanMessage(content="I like tomatoes, what should I eat?")
#     ]
# ))

# print(chat(
#     [
#         SystemMessage(content="You are a nice AI bot that helps a user figure out where to travel in one short sentence"),
#         HumanMessage(content="I like the beaches, where should I go?"),
#         AIMessage(content="You should go to Nice, France"),
#         HumanMessage(content="What else should I do when I'm there?")
#     ]
# ))

# Document
# An object that holds a piece of text AND metadata
# Can filter documents by metadata when making an application and request it to only look at certain documents.
from langchain.schema import Document

print(Document(page_content="This is my document. It is full of text that I've gathered from other places.", 
        metadata={
            'my_document_id' : 234234,
            'my_document_source' : "The LangChain Papers",
            'my_document_create_time' : 1680013019
        }))