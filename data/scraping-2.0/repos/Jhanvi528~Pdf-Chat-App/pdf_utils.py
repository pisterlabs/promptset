# pdf_utils.py
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
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

def get_text_from_any_pdf(pdf_bytes):
    images = convert_pdf_to_img(pdf_bytes)
    final_text = ""
    for pg, img in enumerate(images):
        final_text += convert_image_to_text(img)
    return final_text


# Helper function to convert PDF to images

def convert_pdf_to_img(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    return images

# Helper function to convert image to text using Tesseract OCR

def convert_image_to_text(img):
    text = pytesseract.image_to_string(img)
    return text

# Main function to extract text from a PDF file

def pdf_to_text(pdf_bytes):
    return get_text_from_any_pdf(pdf_bytes)


# download pdf from link

def download_pdf_from_url(url):
    response = requests.get(url)
    file = open("myfile.pdf", "wb")
    file.write(response.content)
    file.close()
    return BytesIO(response.content)