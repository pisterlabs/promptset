from config import get_OpenAI
from openai import OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")


# LLMs
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.6, model=llm_model) # temperature=0.7 is default

st.title('Memory Test:')
st.write(llm.predict("Hello, my name is John. What is your name?"))
st.write(llm.predict("Great, what is my name?")) # should be John - No Access to Memory at this point

#Memory
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

st.title('Conversation Test:')
st.write(conversation.predict(input="Hello, my name is John. What is your name?"))
st.write(conversation.predict(input="Great, what is my name?")) # should be John - No Access to Memory at this point
st.write(memory.load_memory_variables({})) # {} should be empty - Access to Memory at this point

