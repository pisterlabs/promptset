import os 
from apikeys import OPENAI_API_KEY

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

st.title('Study 🧠🧠🧠')
st.divider()

st.caption('An app for document analysis, research and note-taking.')
prompt = st.text_input('Ask anything...')


llm = OpenAI(temperature=0.9)

if prompt:
    response = llm(prompt)
    st.write(response)



