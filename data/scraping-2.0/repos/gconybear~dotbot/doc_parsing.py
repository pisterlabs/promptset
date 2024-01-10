import re
import docx2txt
from pypdf import PdfReader
import streamlit as st  
import mimetypes 
from io import BytesIO
from typing import Any, Dict, List 
from langchain.text_splitter import RecursiveCharacterTextSplitter 


def check_file_type(file):
    file_type, encoding = mimetypes.guess_type(file.name)
    return file_type


@st.cache_data()
def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


@st.cache_data()
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output


@st.cache_data()
def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def split_docs(txt, min_length=100):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        separators=["NSECTION", "\n\n", "\n", ".", "!", "?", ",", " ", ""],
        chunk_overlap=0,
    ) 
    
    chunks = text_splitter.split_text(txt) 
    
    return [c for c in chunks if len(c) > min_length]
    
    