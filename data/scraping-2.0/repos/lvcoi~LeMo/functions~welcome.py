import os
import time
import random
import sys
import re
from dotenv import find_dotenv, load_dotenv
from functions.dictionaries import (onboard_questions, law_keywords)
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
import streamlit as st

user_text = st.empty()

def identify_law_type(user_input, law_keywords):
    user_input_str = str(user_input)  # Convert to string
    user_input_words = re.findall(r'\w+', user_input_str, re.UNICODE)
    user_input_set = set(word.casefold() for word in user_input_words)
    for law_type, keywords in law_keywords.items():
        if any(keyword.casefold() in user_input_set for keyword in keywords):
            return onboard_questions[law_type], law_type
    return None, None

                  

load_dotenv(find_dotenv())
# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Two models are available: fast and smart 
fast = "gpt-3.5-turbo"
smart = "gpt-4-0613"


# Welcome Message
welcome_message = """
Hi there!

I'm Lemo - your AI Paralegal ChatBot, here to help make your legal journey a little less daunting. I understand that navigating legal processes can be complex and often overwhelming, but please know that you're not alone. I'm here to provide as much information and guidance as I can, answering your questions to the best of my abilities.

Remember, no question is too small or too big for me. I'm here to assist 24/7, ready to support you in every step of your journey. While I might not replace the expertise of a human attorney, I will do my utmost to ensure you're well-informed and confident.

Let's embark on this journey together.
"""
# create chatbot instances and set model name and temperature

# create chatbot instance for gpt 4 as smart
smart = ChatOpenAI(
  model_name=fast,  # type: ignore
  temperature=0,
  openai_api_key=OPENAI_API_KEY,
) # type: ignore

# create chatbot instance for gpt 3.5 turbo
fast = ChatOpenAI(
  model_name=fast,  # type: ignore
  temperature=0, 
  openai_api_key=OPENAI_API_KEY,
) # type: ignore

# create conversation chain
conversationBuf = ConversationChain(
  llm=smart, 
  memory=ConversationBufferMemory(), 
  verbose=True,
)

# create entity memory chain
conversationEnt = ConversationChain(
    llm=fast, 
    verbose=True,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    memory=ConversationEntityMemory(llm=fast)
)

user_input = st.empty()

onboard, law_type = identify_law_type(user_input, law_keywords) # type: ignore

# Create message instance
messages = [
  SystemMessage(content="Your name is Lemo. You will be a top notch Paralegal today. Help with {client} onboarding of their {law_type}, collecting {client}'s story, and help format a legal complaint or motion to file with the court. You are working with and under the supervision of {attorney} Do your very best to find the best and most current information, presented in a way thats understandable for the layman and powerful for the court. You are eternally empathetic to the user. You are on their side"),
]

# Memory for conversation

# Create memory instance for Buffered Memory
def cbm_loop(query):
  while True:
    ai_response = conversation.predict(query) # type: ignore
    print(ai_response)

# Create memory instance for Entity Memory
def cem_loop(query):
  while True:
    ai_response = conversation.predict(query) # type: ignore
    print(ai_response)

# Check for OPENAI_API_KEY in .env file. raise error if not found
def api_key_check():
  if OPENAI_API_KEY == None or OPENAI_API_KEY == "":
    raise Exception("OPENAI_API_KEY not found in .env file. Please add it to .env file and try again") 

def fake_type(words):
    words += "\n"
    for char in words:
        time.sleep(random.choice([
          0.02, 0.04, 0.08, 0.014, 0.018,
          0.06, 0.02, 0.04, 0.02, 0.04
        ]))
        sys.stdout.write(char)
        sys.stdout.flush()
    time.sleep(1)

if __name__ == '__main__':
    None # type: ignore