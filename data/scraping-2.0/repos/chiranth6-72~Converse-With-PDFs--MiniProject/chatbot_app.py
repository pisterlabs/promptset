import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")


def get_file_size(file):
    # file.seek(0, os.SEEK_END)
    # file_size = file.tell()
    # file.seek(0)
    # return file_size
    return file.getbuffer().nbytes


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         file_like_object = BytesIO(pdf)
#         pdf_reader = PdfReader(file_like_object)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain


def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(reversed(st.session_state.chatHistory)):
        if i % 2 == 0:
            st.write("Human🗣️:")
            st.write(f"\t{message.content}")
        else:
            st.write("PDFbot🤖:")
            st.write(f"\t{message.content}")

    pass


def main():
    st.set_page_config("Converse with PDFs", page_icon=':books:')
    st.header("Converse with Multiple PDFs 📑💬")
    user_question = st.text_input("Input a query for the PDF Files")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Process Button", accept_multiple_files=True, type=['pdf'])

        # pdf_bytes = None
        if st.button("Process"):
            with st.spinner("Processing"):
                # pdf_bytes = pdf_docs.read()
                # raw_text = get_pdf_text(pdf_bytes)
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                st.session_state.conversation = get_conversational_chain(
                    vector_store)
                st.success("Done")

        # if pdf_bytes is not None:
        #     file_details = {
        #         "Filename": pdf_docs.name,
        #         "File size": f'{len(pdf_bytes)} bytes'
        #     }
        #     st.markdown("<h4 style color:black;'>File details</h4>",
        #                 unsafe_allow_html=True)
        #     st.json(file_details)

        # if pdf_docs is not None:
        #     for uploaded_file in pdf_docs:
        #         file_details = {
        #             "Filename": uploaded_file.name,
        #             "File size": get_file_size(uploaded_file)  # File size is calculated here
        #         }
        #         st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
        #         st.json(file_details)


if __name__ == "__main__":
    main()
