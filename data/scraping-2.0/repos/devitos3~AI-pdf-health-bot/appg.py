# Importing modules
import os 
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message 



# Set API keys and the models to use
API_KEY = "your api key"
model_id = "gpt-3.5-turbo"

# Adding  openai api key for use
os.environ["OPENAI_API_KEY"] = API_KEY

# Loading PDF document with the help of langchain
loaders = PyPDFLoader('/Users/devanshijajodia/Downloads/ADR11.pdf')

# Creating  a vector representation of this document loaded
index = VectorstoreIndexCreator().from_loaders([loaders])

# Setup streamlit app

# Display the page title and the text box for the user to ask the question
st.title(' 💊 Query your PDF document ')
prompt = st.text_input("Enter your question to query your PDF documents")

# Display the current response. No chat history is maintained

if prompt:
 
 response = index.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt, chain_type = 'stuff')

 # Write the results from the LLM to the UI
 st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )

 if prompt:
 
    response = index.query(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.2), question = prompt, chain_type = 'stuff')

 message(prompt, is_user=True)
 message(response,is_user=False )