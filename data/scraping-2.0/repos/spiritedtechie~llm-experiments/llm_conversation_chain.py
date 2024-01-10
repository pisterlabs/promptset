from dotenv import load_dotenv
import os

load_dotenv('.env')

from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback


# model_name = gpt-3.5-turbo, text-davinci-003
# llm = OpenAI(model_name="text-davinci-003", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))

# 
# Conversation chain
#
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

with get_openai_callback() as cb:
    conversation.predict(input="Tell me about yourself.")
    conversation.predict(input="What can you do?")
    conversation.predict(input="What can you tell me about the book 100 years of solitude")
    conversation.predict(input="What else can you tell me about the book")
    conversation.predict(input="Thanks for your help")
    print(cb)

