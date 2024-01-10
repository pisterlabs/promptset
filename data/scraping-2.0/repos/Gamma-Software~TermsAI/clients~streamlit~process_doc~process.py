from typing import List
import re

from langchain.docstore.document import Document
from langchain.schema.vectorstore import VectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from process_doc.utils import (
    extract_text_from_searchable_pdf,
    convert_pdf_to_searchable,
    get_pdf_number_pages,
    extract_text_from_jpeg,
)


def extract_clean_doc(data) -> List[Document]:
    """
    This function extract the text from any form of documents and cleans it.
    It keeps maximum 2 consecutive new lines and split the document into paragraphs.
    """
    if not data or "text" not in data and "pdf" not in data and "pic" not in data:
        raise ValueError("No data provided")

    paragraph_splitter = CharacterTextSplitter(
        separator="\n\n",
        keep_separator=True,
        add_start_index=True,
        chunk_size=1000,
        chunk_overlap=0,
    )
    extracted_docs: List[Document] = []
    remove_extra_newlines = re.compile(r"\n{3,}", re.MULTILINE)

    if "pdf" in data:
        # OCR the pdf to make it searchable if necessary
        searchable = convert_pdf_to_searchable(data["pdf"], data["pdf"])
        total_pages = get_pdf_number_pages(data["pdf"])
        for page_id, is_searchable in searchable:
            content = extract_text_from_searchable_pdf(data["pdf"], page_id)
            extracted_docs.append(
                Document(
                    page_content=remove_extra_newlines.sub("\n\n", content),
                    metadata={
                        "type": "pdf",
                        "page": page_id,
                        "ocr": not is_searchable,
                        "total_pages": total_pages,
                    },
                )
            )
    if "pic" in data:
        extracted_docs.append(
            Document(
                page_content=remove_extra_newlines.sub(
                    "\n\n", extract_text_from_jpeg(data["pic"])
                ),
                metadata={
                    "type": "pic",
                },
            )
        )
    if "text" in data:
        clean_text = remove_extra_newlines.sub("\n\n", data["text"])
        extracted_docs = [Document(page_content=clean_text, metadata={"type": "text"})]

    # Split the document into paragraphs
    extracted_docs = paragraph_splitter.split_documents(extracted_docs)
    return extracted_docs


def embed_doc(filename, docs: List[Document]) -> VectorStoreRetriever:
    embeddings = OpenAIEmbeddings()
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    docsearch = Chroma.from_texts(
        texts, embeddings, metadatas=metadatas, collection_name=filename
    ).as_retriever()
    return docsearch
