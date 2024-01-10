import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
import os
from pathlib import Path

load_dotenv()

with st.sidebar:
    st.title('💬 LLM chatbot 📄')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    "[View source code](https://github.com/Timothy1102/company-assistant-bot)"

def save_file_to_folder(uploadedFile):
    save_folder = 'storage'
    save_path = Path(save_folder, uploadedFile.name)
    with open(save_path, mode='wb') as w:
        w.write(uploadedFile.getvalue())

def delete_file_from_folder(file_name):
    save_folder = 'storage'
    file_path = Path(save_folder, file_name)
    if file_path.exists():
        os.remove(file_path)

def generate_response(uploaded_file, query_text):
    if uploaded_file is not None:
        file_path = os.path.join('storage/', uploaded_file.name)
        documents = []
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split()
        if uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
            documents = loader.load()
        if uploaded_file.name.endswith(".docx") or uploaded_file.name.endswith(".doc"):
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        if uploaded_file.name.endswith(".csv"):
            loader = CSVLoader(file_path)
            documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings()
        # Create a vector store from documents
        db = FAISS.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        # Create QA chain
        llm = OpenAI(temperature=0.4)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)
        return qa.run(query_text)
 
def main():
    st.header("💬 Document Q&A chatbot 📄")
 
    # upload a file
    uploaded_file = st.file_uploader("Upload your file", type=["pdf", "txt", "csv", "doc", "docx"]) # TODO: support more file types
 
    # Accept user questions/query
    query = st.text_input("Ask questions about your file:")

    # TODO: optimize performance
    if query and uploaded_file:
        save_file_to_folder(uploaded_file)
        response = generate_response(uploaded_file, query)
        st.write(response)
        delete_file_from_folder(uploaded_file.name)
 
if __name__ == '__main__':
    main()
