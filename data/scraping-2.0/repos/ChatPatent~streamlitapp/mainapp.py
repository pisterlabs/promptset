# Imports
import os 
#import chromadb
import pypdf
import tiktoken
import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

API_KEY = os.environ.get('OPENAI_API_KEY')
#os.environ["OPENAI_API_KEY"] = API_KEY
model_id = "gpt-3.5-turbo"

llm=ChatOpenAI(model_name = model_id, temperature=0.2)

loaders = PyPDFLoader('docs/pdf.pdf')

st.title('✨ AI Smart Query | AI智慧畅聊 ')
prompt = st.text_input("Welcome to AI Smart Query, questions will be answered based on the sources in our docs directory.\n\n欢迎来到AI智慧畅聊，基于本项目docs目录下的文档资源进行高效QA问答。")

if prompt!="" and not prompt.strip().isspace() and not prompt == "" and not prompt.strip() == "" and not prompt.isspace():    
    index = VectorstoreIndexCreator().from_loaders([loaders])
    # stuff chain type sends all the relevant text chunks from the document to LLM    
    response = index.query(llm=llm, question = prompt, chain_type = 'stuff')    
    # Write the results from the LLM to the UI
    st.write("<br><i>" + response + "</i><hr>", unsafe_allow_html=True )
    #st.write("<b>" + prompt + "</b><br><i>" + response + "</i><hr>", unsafe_allow_html=True )
else:
    st.write("Enter your query first.")
