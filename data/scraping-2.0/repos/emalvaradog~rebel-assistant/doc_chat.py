# Imports
# __import__('pysqlite3')
import os
import sys
import streamlit as st
from tempfile import NamedTemporaryFile
from langchain.chat_models import ChatOpenAI
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator

# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "messages" not in st.session_state:
  st.session_state['messages'] = []

# Streamlit APP
st.header("Rebel by datarebels")
uploaded_file = st.file_uploader("Upload your .pdf file", type="pdf")

# GUI Chat messages
for message in st.session_state.messages:
  if isinstance(message, SystemMessage):
    continue
  elif isinstance(message, HumanMessage):
    with st.chat_message("user"):
      st.write(message.content)
  elif isinstance(message, AIMessage):
    with st.chat_message("rebel"):
      st.write(message.content)
  

user_input = st.chat_input(placeholder="Primero sube un archivo PDF", disabled=True)

if uploaded_file is not None:
    
  user_input = st.chat_input(placeholder="En que te puedo ayudar?", disabled=False)

  with NamedTemporaryFile(delete=False) as tmp_file:
    tmp_file.write(uploaded_file.read())
    tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    index = VectorstoreIndexCreator().from_loaders([loader])
    

    if user_input:
      with st.chat_message("user"):
        st.write(user_input)
      
      st.session_state.messages.append(HumanMessage(content=user_input))

      # res = chat(st.session_state.messages).content
      res = index.query(user_input)

      with st.chat_message("rebel"):
        st.write(res)

      st.session_state.messages.append(AIMessage(content=res))



