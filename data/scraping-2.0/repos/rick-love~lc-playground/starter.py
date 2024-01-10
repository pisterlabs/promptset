from config import get_OpenAI
from openai import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")


# LLMs
llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
chat_model = ChatOpenAI(temperature=0) # temperature=0.7 is default




############
# Show to the screen
# App Framework
st.title('Boiler Plate:')