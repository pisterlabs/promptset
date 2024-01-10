import os
import re
import glob
import shutil
import tabula
import requests
import pandas as pd
import streamlit as st
from typing import List
from PyPDF2 import PdfReader
from datetime import datetime
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text


from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


import Config_Paremeters
from Config_Streamlit import *
from dotenv import load_dotenv
load_dotenv()


upload_data_directory = "table_uploaded"
clean_data_directory = "table_cleaned"


# def extract_tables_from_pdfs(pdf_directory, csv_directory):
#     try:
#         print('\nExtracting tables from PDF files ...')
#         os.makedirs(csv_directory, exist_ok=True)
#         pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
#         for pdf_file in pdf_files:
#             pdf_file_path = os.path.join(pdf_directory, pdf_file)
#             tables = tabula.read_pdf(pdf_file_path, pages='all', multiple_tables=True)
#             for i, table in enumerate(tables):
#                 # csv_file_path = os.path.join(csv_directory, f"{pdf_file.replace('.pdf', '')}_table_{i}.csv")
#                 table.to_csv(f"table{i}.csv")
#         print(f"Tables extracted from {len(pdf_files)} PDF files ... Done")
#     except Exception as e:
#         print(f"Failed to process file {pdf_file_path}: {e}")

def extract_tables_from_pdfs(pdf_directory, csv_directory):
    try:
        print('\nExtracting tables from PDF files ...')
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        for i, pdf_file in enumerate(pdf_files):
            pdf_file_path = os.path.join(pdf_directory, pdf_file)
            tabula.convert_into(pdf_file_path, f"table{i}.csv", output_format="csv", pages="all")
        print(f"Tables extracted from {len(pdf_files)} PDF files ... Done")
    except Exception as e:
        print(f"Failed to process file {pdf_file_path}: {e}")

extract_tables_from_pdfs(upload_data_directory, clean_data_directory)