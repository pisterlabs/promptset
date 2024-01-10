import streamlit as st
from configEnv import settings
from htmlTemplates import css, bot_template, user_template
from pdf_utils import download_pdf_from_url, pdf_to_text
from vectorstore_utils import get_vectorstore, get_conversation_chain, handle_userinput
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
import streamlit as st
import os
import pickle
import warnings
from configEnv import settings
from htmlTemplates import css, bot_template, user_template
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import time
import io
import glob
from pdf2image import convert_from_bytes
from pytesseract import image_to_string
import requests
from io import BytesIO

st.set_page_config(page_title="LegalAI Insight", page_icon="🤖", layout="centered")

def run_streamlit_app():
    st.set_option('deprecation.showfileUploaderEncoding', False)
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, module="streamlit")

    os.environ["OPENAI_API_KEY"] = settings.KEY
    st.write(css, unsafe_allow_html=True)

    st.session_state["conversation"] = None
    st.session_state["chat_history"] = None
    if "session_state" not in st.session_state:
        st.session_state["session_state"] = None

    if st.button("Reload page"):
        st.cache_resource.clear()
        st.session_state["conversation"] = None
        st.session_state["chat_history"] = None
        st.session_state["session_state"] = None
        st.experimental_rerun()

    st.title('🤖 LegalAI Insight')
    pdf_url = st.text_input("Enter PDF URL:")
    pdf_multiple = st.file_uploader("Upload your Pdf", type='pdf',
                           accept_multiple_files=True)
    raw_text = ''
    pdf = []
    if(len(pdf_url) > 0 ):
        pdf = download_pdf_from_url(pdf_url)
        st.write("PDF Loaded!")
        pdf =["myfile.pdf"]
    pdf.extend(pdf_multiple)
    if pdf is not None:
        for single_pdf in pdf:
            if(isinstance(single_pdf, str)):
                pdfreader = PdfReader(single_pdf)
                for i, page in enumerate(pdfreader.pages):
                    content = page.extract_text()
                    if content:
                        raw_text += content
            else:
                pdf_bytes = single_pdf.read()
                raw_text += pdf_to_text(pdf_bytes)

    if 'raw_text' in locals():
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)


    if len(texts) > 0:
        doc_search = get_vectorstore(texts, pdf)
        st.session_state["conversation"] = get_conversation_chain(doc_search)

    query = st.text_input("Ask questions about Pdf file:")
    if query:
        if len(texts) > 0:
            handle_userinput(query)
        else:
            st.write(
                'No data extracted from pdf uploaded. Please upload a correct pdf.')

