from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
import textract
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()
from PIL import Image

# img = Image.open(r"streamlit_apps\understanding\logo.png")
st.set_page_config(page_title="LegalEase: Understand with Ease")
st.header("LegalEase: Understand with Ease📄")
file = st.file_uploader("Upload your file")

chat_history_und = []

if file is not None:
    content = file.read()  # Read the file content once

    if file.type == 'application/pdf':
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif file.type == 'text/plain':
        text = content.decode('utf-8')
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        text = textract.process(content)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )  

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    query = st.text_input("Ask you question")
    if query:
        docs = knowledge_base.similarity_search(query)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        
        chat_history_und.append((query, response))

        # Chat history
        st.text("Conversation History:")
        for turn in chat_history_und:
            st.text(f"User: {turn[0]}")
            st.text(f"LegalEase: {turn[1]}")
            st.text("-----")


           
        st.success(response)