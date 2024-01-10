import streamlit as st
import numpy as np
import pandas as pd
import time
from langchain.memory import ConversationBufferMemory
from dataprocess import return_answer, load_vectorstore
from dotenv import load_dotenv
import openai
import os
from tempfile import NamedTemporaryFile

load_dotenv()

openai.api_key  = os.environ['OPENAI_API_KEY']

openai_models = ["gpt-3.5-turbo", "gpt-4"]


st.header("Chat With Your PDF", anchor=False, divider="rainbow", )


openai_models = ["gpt-3.5-turbo", "gpt-4"]

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", max_len=10, return_messages=True)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    
    st.title('ChatGPT Configuration')

    if 'model' not in st.session_state:
        st.session_state.model = openai_models[0]
    model = st.selectbox('Which OpenAI model would you like to use?',openai_models, index=openai_models.index(st.session_state.model))

    
    # Check session state first
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.0
    temperature = st.slider('Higher temperature more creativity', min_value=0.0, max_value=2.0, step=0.01, value=st.session_state.temperature)
    

    
    file = st.file_uploader(label="Choose a PDF file")
    if file:
        if st.button("Process"):
            with st.spinner("Processing"):
                if file is not None:
                    # Save the file to a temporary file if it's not already saved
                    if 'temp_file_path' not in st.session_state or st.session_state.file != file:
                        st.session_state.file = file
                        # st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", max_len=10, return_messages=True)
                        with NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
                            tmpfile.write(file.getvalue())
                            st.session_state.temp_file_path = tmpfile.name

                        # Initialize retriever only if it's changed/newly uploaded 
                        st.session_state.retriever = load_vectorstore(st.session_state.temp_file_path)

                        st.session_state.qa = return_answer(temperature, model, st.session_state.memory, st.session_state.retriever)

                        os.unlink(st.session_state.temp_file_path)  
    

query = st.chat_input(placeholder="Send a message...")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# react to user input
if 'qa' in st.session_state and query:

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)
    # Add user message to chat history
    st.session_state.messages.append({"role":"user","content": query})

    

    # Call the QA function with the necessary parameters
    result = st.session_state.qa({"question": query})
    # Store the result and answer in the session state
    st.session_state.result = result
    st.session_state.answer = result['answer']
    
    
    # Dsiplay assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = st.session_state.answer
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant message to chat history
    st.session_state.messages.append({"role":"assistant", "content": full_response})



    
    
            


