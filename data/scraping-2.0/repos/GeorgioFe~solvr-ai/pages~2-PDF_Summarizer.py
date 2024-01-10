'''
Author: Georgio Feghali
Date: July 11 2023
'''

# UI Dependencies.
import streamlit as st
from PIL import Image

# Logic Dependencies.
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
import tempfile
import os

# Page Configuration.
favicon = Image.open("./admin/branding/logos/favicon-32x32.png")
st.set_page_config(
    page_title="Solvr.ai - PDF Summarizer",
    page_icon=favicon,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Logic
# Convert file-like object to a temporary file path.
def create_temp_file(file_like_object):
    # Step 1: Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

        # Step 2: Write the contents of the file-like object to the temporary file
        with open(temp_file_path, 'wb') as temp_file_writer:
            temp_file_writer.write(file_like_object.read())

    # Step 3: Return the path to the temporary file
    return temp_file_path

# Loads and processes the document to be used by the chain.
def document_loader(file):
    file_path = create_temp_file(file)
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    return docs

# Setting up chain for summarizing.
def chain_setup():
    OPENAI_API_KEY = st.secrets["openai_api_key"]
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=False)

    return chain

# Gets summary.
def get_summary(document):
    chain = chain_setup()
    docs = document_loader(document)
    answer = chain.run(docs[:])

    return answer

# UI
st.markdown("<h1 style='text-align: center;'>PDF Summarizer 📄</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a PDF file", "pdf")
if uploaded_file is not None:
    if st.button('Summmarize!'):
        with st.chat_message("assistant"):
            with st.spinner('Summarizing PDF...'):
                summary = get_summary(uploaded_file)
            st.write("Here is the summary of the provided PDF!")
            st.markdown(summary)