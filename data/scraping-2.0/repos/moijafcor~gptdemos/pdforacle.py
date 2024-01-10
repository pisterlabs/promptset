# pdforacle.py

from bin.alejandroao_html_chatbot_template import css, bot_template, user_template
import dotenv
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_experimental.pal_chain.base import PALChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub

# from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import (
    # Role the Bot should act as
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain.vectorstores import FAISS
import os
from PyPDF2 import PdfReader
import streamlit as st
from streamlit_chat import message
import sys


def scaffold() -> None:
    st.set_page_config(
        page_title="PDF Oracle | Inspired on ChatGPT", page_icon=":grin:"
    )
    st.write(css, unsafe_allow_html=True)


def get_pdf_stream(pdf_files):
    stream = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for chunk in pdf_reader.pages:
            stream += chunk.extract_text()
    return stream


def chunker(stream):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_text(stream)


def vectordb(chunk):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    return FAISS.from_texts(texts=chunk, embedding=embeddings)


def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.5, "max_length": 512},
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )


def user_input_handler(user_input):
    response = st.session_state.conversation({"question": user_input})
    # st.write(response)
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )


def main() -> None:
    scaffold()
    dotenv.load_dotenv()

    # message("Hello!")
    # message("All good", is_user=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        st.subheader("Your PDFs Collection")
        pdf_raw = st.file_uploader(
            "Please upload PDFs and hit 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Working"):
                pdf_stream = get_pdf_stream(pdf_raw)
                chunks = chunker(pdf_stream)
                # st.write(chunks)
                vectorstore = vectordb(chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

    st.header("PDF Oracle :grin:")
    user_input = st.text_input("Query your PDFs:")
    if user_input:
        user_input_handler(user_input)

    st.write(
        user_template.replace("{{MSG}}", "Greetings, oracle!"),
        unsafe_allow_html=True,
    )
    st.write(
        bot_template.replace("{{MSG}}", "Hello, earthling..."),
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
