import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import pickle
from pathlib import Path
from langchain.storage import LocalFileStore


# the knowledge folder under thesame directory as this file
DOC_DIR = os.path.join(os.path.dirname(__file__), "knowledge")

def get_pdf_text(pdf_docs):
    data = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for (idx, page) in enumerate(pdf_reader.pages):
            data.append({
                "meta": {
                    "file": os.path.basename(pdf.name),
                    "page": idx + 1,
                    "source": "{file}, page {page}".format(file=os.path.basename(pdf.name), page=idx + 1),
                },
                "text": page.extract_text()
            })
    return data

def get_text_chunks(data):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=200,
        chunk_overlap=20
    )

    return_data = [d.copy() for d in data]

    for page, return_page in zip(data, return_data):
        tmp_chunks = text_splitter.split_text(page['text'])
        new_chunks = ["[" + Path(page["meta"]["file"]).stem + "] " + chunk for chunk in tmp_chunks]
        return_page["chunks"] = new_chunks

    return return_data

# TODO PRIORITY-LOW
# TODO Batch the embeddings calls
# TODO Use a local embeddings model
def get_vectorstore(pgbar, text_chunks):
    embeddings = OpenAIEmbeddings()
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        embeddings, LocalFileStore("./cache/"), namespace=embeddings.model
    )
    vectorstores = FAISS.from_texts(["empty"], cached_embedder, metadatas=[text_chunks[0]["meta"]])

    # Ensure the cache directory exists
    if not os.path.exists("cache"):
        os.makedirs("cache")

    # TODO Only build VS for one time
    for (idx, page) in enumerate(text_chunks):
        pgbar.progress(idx / len(text_chunks),
                       text=f"Processing page {idx + 1} of {len(text_chunks)}")
        # Check if embeddings file exists for the page
        vectorstore = FAISS.from_texts(texts=page["chunks"],
                                embedding=cached_embedder, 
                                metadatas=[page["meta"] for _ in page["chunks"]])

        vectorstores.merge_from(vectorstore)

    # print(vectorstores.docstore._dict)
    pgbar.empty()

    return vectorstores

@st.cache_resource
def prepare_pdfs():
    pgbar = st.progress(0, "Preparing Knowledges...")
    pdfs = [f for f in os.listdir(DOC_DIR) if f.endswith(".pdf")]
    handles = [open(os.path.join(DOC_DIR, pdf), "rb") for pdf in pdfs]
    data = get_pdf_text(handles)
    # Close the files
    for pdf in handles:
        pdf.close()

    # get the text chunks
    text_chunks = get_text_chunks(data)

    # create vector store
    vectorstore = get_vectorstore(pgbar, text_chunks)

    return vectorstore