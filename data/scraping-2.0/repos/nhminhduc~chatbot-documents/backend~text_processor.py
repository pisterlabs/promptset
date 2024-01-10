from typing import Set, List
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.utils import (
    read_csv,
    read_docx,
    read_html,
    read_pdf,
    read_pptx,
    read_txt,
    read_xlsx,
    read_xml,
    read_json,
)


def parse_file(file: UploadedFile) -> tuple[str, List[str]]:
    file_type = file.type
    file_name = file.name
    documents = []
    if file_type == "application/pdf":
        documents += read_pdf(file)
    elif (
        file_type
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        documents += read_docx(file)
    elif file_type == "text/plain":
        documents += read_txt(file)
    elif (
        file_type
        == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ):
        documents += read_pptx(file)
    elif (
        file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        documents += read_xlsx(file)
    elif file_type == "text/csv":
        documents += read_csv(file)
    elif file_type == "text/html":
        documents += read_html(file)
    elif file_type == "application/xml":
        documents += read_xml(file)
    elif file_type == "application/json":
        documents += read_json(file)
    return file_name, documents


def text_to_docs(text: tuple) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, (str, tuple)):
        # Take a single string as one page
        name = text[0]
        content = text[1]
    page_docs = [Document(page_content=page) for page in content]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{name}-{doc.metadata['page']}"
            doc_chunks.append(doc)

    return doc_chunks


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "Sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string
