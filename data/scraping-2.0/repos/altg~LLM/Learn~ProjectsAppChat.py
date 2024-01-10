# Streamlit App to analyse Projects using GPT

import pandas as pd
import streamlit as st

import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.document_loaders import DataFrameLoader

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


st.title("Project Analysis - GPT")

# Load Embeddings DB

@st.cache_resource
def load_embeddings():
    embeddings = OpenAIEmbeddings()

    embed = st.text("Loading Data Embeddings .....")
    store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings, collection_name='projects' )

    embed.text("Loading Data Embeddings ..... Done ! ")

    return store


store = load_embeddings()

retriever = store.as_retriever()

# Setup LLM and QA Chain

llm = ChatOpenAI(temperature = 0.0, model_name="gpt-3.5-turbo-16k-0613")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What's your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run the QA Chain and get the response
    response = qa_chain.run(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



#"Session Sate Object:" , st.session_state
