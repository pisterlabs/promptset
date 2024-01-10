import streamlit as st
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
import os 
import constants
from dotenv import load_dotenv
load_dotenv()


os.environ['OPENAI_API_KEY'] = constants.OPENAI_API_KEY

st.title("Kiki's Chatbot")
question = st.text_input("Ask me anything:")
loader = TextLoader("about_kiki.txt")
index = VectorstoreIndexCreator().from_loaders([loader])

if st.button('Generate'):
    if question:
        with st.spinner('Generating response...'):
            response = index.query(question)
            st.write(response)
    else:
        st.warning('Please enter your prompt')



