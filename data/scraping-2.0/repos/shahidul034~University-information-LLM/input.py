from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
import os

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

def save_pdf_to_data(pdf_content, file_name="user_uploaded.pdf"):
    # Save the PDF in the data/ directory
    pdf_path = os.path.join(DATA_PATH, file_name)
    with open(pdf_path, "wb") as pdf_file:
        pdf_file.write(pdf_content)
    loader=DirectoryLoader(DATA_PATH,glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts=text_splitter.split_documents(documents)

    embeddings= HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
    model_kwargs ={ 'device': 'cpu' }
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(DB_FAISS_PATH)


def main():
    st.header("Save PDF to DATA_PATH 💬")

    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_content = pdf.read()
        save_pdf_to_data(pdf_content)

if __name__ == '__main__':
    main()
