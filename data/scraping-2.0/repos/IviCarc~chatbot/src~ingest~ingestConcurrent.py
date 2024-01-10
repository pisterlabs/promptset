import os
import concurrent.futures
from typing import Callable, List, Tuple, Dict
from pdfplumber import open as pdf_open
from PyPDF4 import PdfFileReader
import re
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import threading

db_lock = threading.Lock()


def extract_metadata_from_pdf(file_path: str) -> dict:
    with open(file_path, "rb") as pdf_file:
        reader = PdfFileReader(pdf_file)
        metadata = reader.getDocumentInfo()
        return {
            "title": metadata.get("/Title", "").strip(),
            "author": metadata.get("/Author", "").strip(),
            "creation_date": metadata.get("/CreationDate", "").strip(),
        }


def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with pdf_open(file_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():
                pages.append((page_num + 1, text))
    
    return pages


def parse_pdf(file_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_pages_from_pdf(file_path)

    return pages, metadata


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def process_pdf(file_path: str):
    raw_pages, metadata = parse_pdf(file_path)
    cleaned_text_pdf = [(page_num, clean_text(text)) for page_num, text in raw_pages]
    document_chunks = text_to_docs(cleaned_text_pdf, metadata)
    # document_chunks = document_chunks[:70]  # Limit the number of processed pages

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        collection_name="april-2023-economic",
        persist_directory="src/data/chroma",
    )

    # Adquirir el bloqueo antes de escribir en la base de datos
    with db_lock:
        vector_store.persist()


def text_to_docs(text: List[str], metadata: Dict[str, str]) -> List[Document]:
    doc_chunks = []
    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(page)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-{i}",
                    **metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


if __name__ == "__main__":
    load_dotenv()
    
    pdf_directory = "src/data/pdfs"
    pdf_files = [os.path.join(pdf_directory, pdf) for pdf in os.listdir(pdf_directory)]

    # Utilizar el bloqueo al procesar los archivos PDF
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_pdf, pdf_files)