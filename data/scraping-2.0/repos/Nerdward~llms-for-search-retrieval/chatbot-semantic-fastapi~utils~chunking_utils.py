import hashlib
from datetime import datetime
import os
from tempfile import NamedTemporaryFile
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def clean_newlines(text: str):
    pattern = r'(?<=\S)\n(?=\S)'
    replacement = ' '
    return re.sub(pattern, replacement, text)


def my_hash(s):
    # Return the MD5 hash of the input string as a hexadecimal string
    if isinstance(s, str):
        return hashlib.md5(s.encode()).hexdigest()
    
    return hashlib.md5(s.page_content.encode()).hexdigest()

def prep_documents_for_vector_storage(documents, streamlit=False, chunk_size=500):
    if streamlit:
        bytes_data = documents.read()
    else:
        bytes_data = documents.file.read()
    with NamedTemporaryFile(delete=False) as tmp:  # open a named temporary file
        tmp.write(bytes_data)                      # write data from the uploaded file into it
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0
        )
        documents = PyPDFLoader(tmp.name).load_and_split(text_splitter=text_splitter) 
        # documents = PyPDFLoader(data).load_and_split()       # <---- now it works!
    os.remove(tmp.name)  
        
    ids, texts, metadatas = [], [], []
    for document in documents:
        text, metadata = document.page_content, document.metadata
        ids += [my_hash(text)]
        texts.append(clean_newlines(text)) 
        metadatas.append(metadata)

    return ids, texts, metadatas