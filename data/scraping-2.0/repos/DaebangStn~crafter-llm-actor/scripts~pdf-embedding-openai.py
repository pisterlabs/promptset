import pdfplumber
import PyPDF4
import os
import re
from pathlib import Path
from ruamel.yaml import YAML
from typing import List, Dict, Tuple, Callable

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


def extract_metadata_from_pdf(pdf_path: str) -> dict:
    assert os.path.isfile(pdf_path), f"pdf_path: {pdf_path} must be a valid file path"
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF4.PdfFileReader(pdf_file)
        _metadata = pdf_reader.getDocumentInfo()

        return {
            "title": _metadata.get("/Title", "").strip(),
            "author": _metadata.get("/Author", "").strip(),
            "creation_date": _metadata.get("/CreationDate", "").strip(),
        }


def extract_pages_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    assert os.path.isfile(pdf_path), f"pdf_path: {pdf_path} must be a valid file path"

    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip() is not None:
                pages.append((page_num + 1, text))

    return pages


def parse_pdf(pdf_path: str) -> Tuple[List[Tuple[int, str]], Dict[str, str]]:
    assert os.path.isfile(pdf_path), f"pdf_path: {pdf_path} must be a valid file path"

    _metadata = extract_metadata_from_pdf(pdf_path)
    pages = extract_pages_from_pdf(pdf_path)

    return pages, _metadata


def merge_hyphenated_words(text: str) -> str:
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def fix_newlines(text: str) -> str:
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text: str) -> str:
    return re.sub(r"\n{2,}", "\n", text)


def clean_text(pages: List[Tuple[int, str]], _cleaning_functions: List[Callable[[str], str]]
               ) -> List[Tuple[int, str]]:
    _cleaned_pages = []
    for page_num, text in pages:
        for cleaning_function in _cleaning_functions:
            text = cleaning_function(text)
        _cleaned_pages.append((page_num, text))

    return _cleaned_pages


# convert list of strings to list of documents with metadata
def text_to_docs(text: List[Tuple[int, str]], _metadata: Dict[str, str]) -> List[Document]:
    doc_chunks = []
    for page_num, page in text:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            separators=["\n\n", "\n", ".", "?", "!", " ", ",", ""],
            chunk_overlap=200,
        )
        chunks = text_splitter.split_text(page)

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chunk": i,
                    "source": f"p{page_num}-c{i}",
                    **_metadata,
                },
            )
            doc_chunks.append(doc)

    return doc_chunks


if __name__ == "__main__":
    pdf_dir_path = Path("/res/pdf")
    data_saving_path = Path("C:/Users/geon/PycharmProjects/crafter-llm-actor/data/chroma")
    secrets_path = Path("C:/Users/geon/PycharmProjects/crafter-llm-actor/secrets.yaml")

    assert pdf_dir_path.is_dir(), f"pdf_dir_path: {pdf_dir_path} must be a valid directory path"
    assert data_saving_path.is_dir(), f"data_saving_path: {data_saving_path} must be a valid directory path"
    assert secrets_path.is_file(), f"configs_path: {secrets_path} must be a valid file path"

    yaml = YAML()
    with open(str(secrets_path), "r") as f:
        secrets = yaml.load(f)

    os.environ["OPENAI_API_KEY"] = secrets["api_key"]["openai"]

    pdf_files_path = pdf_dir_path.glob("*.pdf")
    collection_name_list = []

    for pdf_file_path in pdf_files_path:
        pdf_file_path_str = str(pdf_file_path)
        print(f"Processing {pdf_file_path_str}...")
        raw_pages, metadata = parse_pdf(pdf_file_path_str)

        cleaning_functions = [
            merge_hyphenated_words,
            fix_newlines,
            remove_multiple_newlines,
        ]

        cleaned_text_pdf = clean_text(raw_pages, cleaning_functions)
        document_chunks = text_to_docs(cleaned_text_pdf, metadata)

        embeddings = OpenAIEmbeddings()
        collection_name = pdf_file_path.stem.strip().replace(' ', '_')
        collection_name_list.append(collection_name)
        vector_store = Chroma.from_documents(
            document_chunks,
            embeddings,
            collection_name=collection_name,
            persist_directory=str(data_saving_path),
        )

        vector_store.persist()
        print(f"Finished processing {pdf_file_path_str}")

    print(f"Finished processing all pdf files in {pdf_dir_path}")
    print(f"Data saved to {str(data_saving_path)}")
    print(f"Collection names: {collection_name_list}")
