import re
from io import BytesIO
from typing import Tuple, List
import pickle

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import faiss
import os
from pathlib import Path

def parse_pdf(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    try:
        pdf = PdfReader(file)
        output = []
        print(len(pdf.pages))
        for page in pdf.pages:
            text = page.extract_text()
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            text = re.sub(r"\n\s*\n", "\n\n", text)
            output.append(text)

        return output, filename

    except Exception as e:
        # Handle the exception here, you can print an error message or perform other actions.
        print(f"An error occurred: {str(e)}")
        return [], filename

# function to parse text files
def parse_text(file: BytesIO, filename: str) -> Tuple[List[str], str]:
    text = file.getvalue().decode('utf-8')
    if '\n\n\n' in text:
        sections = text.split('\n\n\n')
    else:
        sections = text.split('\n\n')
    return sections, filename

def text_to_docs(text: List[str], filename: str) -> List[Document]:
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text]
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    doc_chunks = []
    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc.metadata["filename"] = filename  # Add filename to metadata
            doc_chunks.append(doc)
    return doc_chunks


def docs_to_index(docs, openai_api_key):
    # embedding model is  text-embedding-ada-002
    # print(docs)
    index = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))
    return index


def get_index_for_files(files, file_names, openai_api_key):
    documents = []
    for file, file_name in zip(files, file_names):
        if file_name.endswith('.pdf'):
            text, filename = parse_pdf(BytesIO(file), file_name)
        else:
            text, filename = parse_text(BytesIO(file), file_name)
        documents = documents + text_to_docs(text, filename)
    index = docs_to_index(documents, openai_api_key)
    return index

def load_predefined_files(folder_path):
    predefined_files = []
    file_names = []
    for file_path in Path(folder_path).glob('*.*'):
        if file_path.suffix in ['.pdf', '.txt']:
            with open(file_path, 'rb') as file:
                predefined_files.append(BytesIO(file.read()))
                file_names.append(file_path.name)
    return predefined_files, file_names