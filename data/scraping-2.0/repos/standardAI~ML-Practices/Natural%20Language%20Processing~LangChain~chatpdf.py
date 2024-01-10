"""
python -m pip install langchain chromadb pypdf pygpt4all pdf2image
Run with this command: streamlit run chatpdf.py
"""

import streamlit as st
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path


PATH = '/home/user-name/.local/share/nomic.ai/GPT4All/nous-hermes-13b.ggmlv3.q4_0.bin'
#PATH = '/home/user-name/.local/share/nomic.ai/GPT4All/ggml-gpt4all-j-v1.3-groovy.bin'

st.set_page_config(page_title="Chat with PDF", page_icon=":robot:")
st.header("Chat with PDF")

uploaded_file = st.file_uploader("Upload Files", type=['pdf'])
user_input = st.text_input("Ask a question")
if uploaded_file and user_input:
    loader = PyPDFLoader(uploaded_file.name)
    documents = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64,
    )
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings, persist_directory='db')
    llm = GPT4All(model=PATH, verbose=False)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        verbose=False,
    )
    images = convert_from_path(uploaded_file.name, 500)
    st.image(images[0])
    answer = qa(user_input)
    st.write(answer['result'])
