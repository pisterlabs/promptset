from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS, Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
import streamlit as st
import os

api_key = st.secrets["openai"]["OPENAI_API_KEY"]


def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=[" ", ",", "\n"]
    )

    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, document):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    try:
        vectorstore = FAISS.load_local(f"{document}_faiss_index", embeddings)
    except Exception as e:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.save_local(f"{document}_faiss_index")
    return vectorstore


def get_conversation_chain(vectorstore, prompt, input_key):
    llm = ChatOpenAI(temperature=0.7, openai_api_key=api_key)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, input_key=input_key
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
    )
    return (conversation_chain, memory)


def get_embeddings(text_data, document):
    chunks = get_text_chunks(text_data)
    vectorStoreObj = get_vectorstore(chunks, document)
    st.session_state["vectorStore"] = vectorStoreObj
