#use dotenv to get the api keys from .env file
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())




#Run basic query with OpenAI wrapper
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003")
# print(llm("explain large language models in one sentence"))




#import schema(structure) for chat messages and ChatOpenAI in order to query chatmodels GPT-3.5-turbo or GPT-4
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.3)
messages = [
    SystemMessage(content="You are an expert data scientist"),
    HumanMessage(content="Write a Python script that trains a neural network on simulated data ")
]
#give human message and system message as an input
response=chat(messages)
# print(response.content,end='\n')
