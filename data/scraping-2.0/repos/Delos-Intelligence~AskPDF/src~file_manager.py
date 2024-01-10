import io

import PyPDF2, tempfile, tiktoken
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from connectors.mongo_connector import store_chunks_in_database

def compute_number_of_tokens(string: str, encoding_name: str = "gpt2") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300,length_function = compute_number_of_tokens)

def split_and_store_chunks(file_info, tempfile_path):
    print("Splitting and storing chunks in chunks database")
    document_chunks = read_and_split(file_info, tempfile_path)
    store_chunks_in_database(file_info, document_chunks)
    return

def read_and_split(file_info, tempfile_path) -> list[dict]:
    document_text = read(file_info, tempfile_path)
    document_chunks = split(document_text)
    return document_chunks

def read(file_info, tempfile_path) -> str:
    with open(tempfile_path, 'rb') as f:
        file_content = f.read()
        if file_info.type == '.pdf':
            document = read_pdf(file_content)
    return document

def read_pdf(file_content : str) -> str:
    pdf_file = io.BytesIO(file_content)
    reader = PyPDF2.PdfReader(pdf_file)
    document = ""
    for page in reader.pages:
        document += page.extract_text()
    return document

def read_pdf_by_page(file):
    reader = PyPDF2.PdfReader(file)
    document = []
    for i,page in enumerate(reader.pages):
        document.append({"page":i, "page_content":page.extract_text()})
    return document

def split(document_text : str) -> list[dict]:
    document_chunks = TEXT_SPLITTER.create_documents([document_text])
    return document_chunks