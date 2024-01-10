import os

from langchain.llms import OpenAI
from langchain.prompts import (PromptTemplate, FewShotPromptTemplate)
from langchain.memory import ConversationBufferMemory 
from langchain.chains import (LLMChain, SimpleSequentialChain, SequentialChain)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from typing_extensions import Concatenate
import streamlit as st
import pandas as pd


st.title('PDF - Question Answering')
from PyPDF2 import PdfReader


# os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
# key_flag = True
key_flag = False
if os.path.exists('./openai_key.py'):
    print("Key file found")
    from openai_key import OPENAI_API_KEY
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    key_flag = True
else:
    key = st.text_input("Enter the openai key")
    if key:
        os.environ['OPENAI_API_KEY'] = key
        key_flag = True

if key_flag:
    embeddings = OpenAIEmbeddings()
    llm = OpenAI(temperature=0.8)
    # read all the pdf files from the dropbox directory
    document_search_space = None
    input_text = st.text_input("Enter the local path to the pdf folder")
    uploaded_file = st.file_uploader("Upload a PDF file")
    myPdfReader = None
    if input_text:
        if os.path.exists(input_text):
            pdf_files = [os.path.join(input_text, f) for f in os.listdir(input_text) if f.endswith('.pdf')]
            st.info(f"Number of pdf files found : {len(pdf_files)}")
            raw_text = ''
            for pdf_file in pdf_files:
                myPdfReader = PdfReader(pdf_file)
                for page in myPdfReader.pages:
                    raw_text += page.extract_text()

            text_splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=800,
                chunk_overlap=150,
                length_function = len,
            )
            texts = text_splitter.split_text(raw_text)
            # print("Length of the chunks: ",len(texts))
            # show the length to the streamlit as log
            st.info(f"Length of the chunks: {len(texts)}")
            st.info("Creating the vector store")
            document_search_space = FAISS.from_texts(texts, embeddings)
            # print("Vector store created")
            st.info("Vector store created")

            uno_chain = load_qa_chain( llm=llm, chain_type="stuff")
            # query = "What did Lencho hope for?"
            query = st.text_input("Enter the question")
            # print(*docs, sep='\n\n')
            if query:
                docs = document_search_space.similarity_search(query,k = 5)
                df = pd.DataFrame([doc.page_content for doc in docs])
                st.write(uno_chain.run(input_documents = docs,question=query))
                with st.expander(f"Top 5 context for the question: {query}"):
                    st.table(df)

    if uploaded_file:
        raw_text = ''
        myPdfReader = PdfReader(uploaded_file)
        for page in myPdfReader.pages:
            raw_text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=800,
            chunk_overlap=150,
            length_function = len,
        )
        texts = text_splitter.split_text(raw_text)
        # print("Length of the chunks: ",len(texts))
        # show the length to the streamlit as log
        st.info(f"Length of the chunks: {len(texts)}")

        st.info("Creating the vector store")
        document_search_space = FAISS.from_texts(texts, embeddings)
        # print("Vector store created")
        st.info("Vector store created")

        uno_chain = load_qa_chain( llm=llm, chain_type="stuff")

        # query = "What did Lencho hope for?"
        query = st.text_input("Enter the question")
        # print(*docs, sep='\n\n')
        if query:
            docs = document_search_space.similarity_search(query,k = 5)
            print(docs[0])
            df = pd.DataFrame([doc.page_content for doc in docs])
            st.write(uno_chain.run(input_documents = docs,question=query))
            with st.expander(f"Top 5 context for the question: {query}"):
                st.table(df)
