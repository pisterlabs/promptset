from langchain.chat_models import ChatVertexAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
from google.cloud import aiplatform
import os
from langchain.llms import VertexAI
from langchain import PromptTemplate, LLMChain
load_dotenv()


st.title("Google Vertex AI PaLM")
model = st.selectbox("Select a model",["codechat-bison", "code-bison", "code-gecko", "text-bison", "textembedding-gecko"])


llm = VertexAI()

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    question = prompt
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        output = llm_chain.run(question)
        st.write(output)