import openai
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredPDFLoader

load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY', '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')

persist_directory = 'ai_paper1'
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

if not os.path.exists(persist_directory):
    print('embedding the document now')
    loader = UnstructuredPDFLoader('ai_paper.pdf', mode="elements")
    pages = loader.load_and_split()

    vectordb = Chroma.from_documents(documents=pages, embedding=embeddings, persist_directory=persist_directory)

    