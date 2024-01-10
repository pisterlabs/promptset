import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
os.environ['OPENAI_API_KEY'] = "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"
os.environ['SERPAPI_API_KEY'] = '360d22e4bc0b06f384cdc79db107bd5ef547daa1c1843698dfcff447654b98e5'

pdf_file = r"D:\project\langchain-openai-chatbot\ReAct.pdf"
pdf_file = pdf_file.replace('\\', '/')
vector_dir = pdf_file.replace('.pdf','')
if not os.path.isdir(vector_dir):
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split(text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0))
    vectorstore  = FAISS.from_documents(pages, OpenAIEmbeddings())
    vectorstore.save_local(vector_dir)
else:
    vectorstore = FAISS.load_local(vector_dir, OpenAIEmbeddings())

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), return_source_documents=True)

# chat_history = []
# query = "这篇论文的工作内容是?"
# result = qa({"question": query, "chat_history": chat_history})
# print(result['answer'])
# print(result['source_documents'][0])
# query = "作者的单位是?"
# result = qa({"question": query})
# print(result['answer'])


