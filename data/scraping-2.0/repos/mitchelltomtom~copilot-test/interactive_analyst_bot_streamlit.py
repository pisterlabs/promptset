#model example from: https://github.com/techleadhd/chatgpt-retrieval/blob/main/README.md
import os
import sys
import openai
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
import constants


#hack to get around sqlite3 error
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#set up api key
os.environ["OPENAI_API_KEY"] = constants.API_KEY

#function to create a vector store from a directory
def createVectorStoreFromDirectory (directory_name: str, read_from_cache: bool):
    if read_from_cache and os.path.exists(directory_name):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory=directory_name, embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
        loader = DirectoryLoader(directory_name)
        if read_from_cache:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])
    print("Index created");
    return index


index = createVectorStoreFromDirectory("real_test_data/", True);

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)


#Create chatbot interface
chat_history = []
st.title('🏛️ Bank Analyst Bot')
openai_api_key = st.sidebar.text_input('OpenAI API Key')
def generate_response(input_text: str, chain_model: function ):
  result = chain_model({"question": input_text, "chat_history": chat_history})
  st.info(result['answer'])

with st.form('my_form'):
  text = st.text_area('Enter text:', 'What are three highlights from 2022 for the bank?')
  submitted = st.form_submit_button('Submit')
  if submitted:
    generate_response(text, chain)