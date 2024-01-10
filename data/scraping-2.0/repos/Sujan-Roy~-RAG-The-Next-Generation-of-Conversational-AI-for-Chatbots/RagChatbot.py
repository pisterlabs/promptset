# import Libraries
from pypdf import PdfReader
import faiss
import streamlit as st
import os
from langchain.vectorstores import chroma
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
import pickle


def parse_pdf(file:BytesIO, filename: str) -> Tuple[list[str],list]:
    """
    Parses a pdf file and returns a list of pages
    """
    pdf = PdfReader(file)
    pages = []

    for page in pdf.pages:
        #extract text from page
        text = page.extract_text()

        # replace word that are spilts by hyphens at the end of the line
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # replace single newlines with spaces but not those flanked by spaces
        text = re.sub(r'(?<! )\n(?![ ])', ' ', text.strip())
        # Consolidate multiple newlines to two newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)

        #append text to pages
        pages.append(text)
    
    #return the cleaned texts and the pages, filename
    return pages, filename


def text_to_docs(text: list[str], filename: str) -> list[Document]:
    # Please note that the input text is a list. If its single text, please convert it to a list

    if isinstance(text, str):
        text = [text]

        # convert each text (page) to a document
    page_docs = [Document(page_content=page) for page in text]

    # assign a page number to each document
    for i, doc in enumerate(page_docs):
        doc.metadata["page_number"] = i + 1
        #doc.page_number = i + 1
    
    doc_chunks = []

    # chunk the documents into smaller chunks and store them in separate documents
    for doc in page_docs:
        # intialize the text spillter with specific chunk size and delimeter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, 
            chunk_overlap=0,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            )
        # split the document into chunks
        chunks=text_splitter.split_text(doc.page_content)

        # convert each chunk to a document and storing its chunk number, page number and filename
        for i, chunk in enumerate(chunks):
            doc= Document(
                page_content=chunk,
                metadata={
                    "page_number": doc.metadata["page_number"], 
                    "chunk_number": i}
            )
            doc.metadata["source_filename"] = f"{doc.metadata['page_number']}-{doc.metadata['chunk_number']}-{filename}"
            doc.metadata["filename"] = filename
            doc_chunks.append(doc)

    return doc_chunks

def docs_to_index(docs,openai_api_key):
    index= FAISS.from_documents(docs,OpenAIEmbeddings(openai_api_key= openai_api_key))
    return index

def get_index_for_pdf(pdf_files,pdf_names,openai_api_key):
    documents=[]
    for pdf_file,pdf_name in zip(pdf_files,pdf_names):
        pages, filename = parse_pdf(BytesIO(pdf_file), pdf_name)
        documents= documents+text_to_docs(pages, filename)
    index= docs_to_index(documents,openai_api_key)
    return index

