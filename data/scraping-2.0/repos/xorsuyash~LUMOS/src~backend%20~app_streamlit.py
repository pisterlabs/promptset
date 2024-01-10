from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import streamlit as st 
from PyPDF2 import PdfReader

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text




def pdf_text(file_path):
    with open(file_path,'r') as f:
        f=f.read()

    
st.header("LUMOS  :books:")
user_question = st.text_input("Ask a question about your subject:")

with st.sidebar:
    st.subheader("Your Documents")
    pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            raw_text = get_pdf_text(pdf_docs)