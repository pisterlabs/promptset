import streamlit as st
import getpass
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load the API key from .env GOOGLE_API_KEY
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Initialize the model
llm = ChatGoogleGenerativeAI(model='gemini-pro')

# Streamlit application
st.title('LangChain Google Generative AI')

# User input
user_input = st.text_input('Enter a message:')

if st.button('Generate Ballad'):
    # Use the model to generate a response
    result = llm.invoke(f"Write a ballad about {user_input}")
    
    # Display the model's response
    st.write(result.content)

if st.button('Generate Limerick'):
    # Use the model to generate a response
    for chunk in llm.stream(f"Write a limerick about {user_input}."):
        st.write(chunk.content)

# System and human message input
system_message = st.text_input('Enter a system message:')
human_message = st.text_input('Enter a human message:')

if st.button('Generate Conversation'):
    model = ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    conversation = model(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]
    )
    st.write(conversation)