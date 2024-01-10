import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pdf2txt import convert_pdfs_to_txt


def get_txt_files_path(dir_path):
    pdf_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".txt"):
                pdf_files.append(os.path.abspath(os.path.join(root, file)))

    return pdf_files


def delete_files(dir_path, file_type=".pdf"):
    for filename in os.listdir(dir_path):
        if filename.endswith(file_type):
            os.remove(os.path.join(dir_path, filename))


def load_documents(uploaded_files_path):
    convert_pdfs_to_txt(uploaded_files_path)
    delete_files(uploaded_files_path)

    txt_path = get_txt_files_path(uploaded_files_path)
    documents = [TextLoader(txt, encoding="utf-8").load() for txt in txt_path]

    return documents


def load_split_documents(uploaded_files_path, chunk_size=14000, chunk_overlap=0):
    documents = load_documents(uploaded_files_path)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    documents_split = [text_splitter.split_documents(doc) for doc in documents]

    return documents_split
