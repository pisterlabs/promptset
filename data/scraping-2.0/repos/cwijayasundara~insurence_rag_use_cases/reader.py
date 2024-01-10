import openai
import os
import streamlit as st

from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

vectorstore = Chroma(persist_directory="./vectorstore",
                     embedding_function=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-4-1106-preview")

output_parser = StrOutputParser()

setup_and_retrieval = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
chain = (setup_and_retrieval
         | prompt
         | model
         | output_parser)

st.subheader("Chat with Insurance Documents:")
request = st.text_area('How can I help you today? ', height=100)
submit = st.button("submit", type="primary")

if request and submit:
    response = chain.invoke(request)
    st.write(response)
