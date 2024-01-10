from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import streamlit as st
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


text =""
os.environ['OPENAI_API_KEY'] = st.secrets['apikey']
chain = load_qa_chain(OpenAI(),chain_type='stuff')
st.set_page_config(page_title='DOCGPT')
st.header('DocGPT :books:')
prompt = st.text_input("Ask a question about your document:")

text_splitter =CharacterTextSplitter(
    separator='\n',
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)


embeddings = OpenAIEmbeddings()

with st.sidebar:
    st.subheader("Your Document")
    pdf_doc = st.file_uploader('Upload your PDF')
    if st.button('Process'):
        with st.spinner("Processing"):
            pdf_reader = PdfReader(pdf_doc)
            for page in pdf_reader.pages:
                text+=page.extract_text()
            texts = text_splitter.split_text(text)
            document_search = FAISS.from_texts(texts,embeddings)
            if prompt:
                docs = document_search.similarity_search(prompt)
                st.write(chain.run(input_documents=docs,question=prompt))