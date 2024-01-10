import json
import re
import time
import logging as log
from typing import List, Dict
from langchain.docstore.document import Document
from langchain.document_transformers import beautiful_soup_transformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.utils import create_documents, document_regex_sub, document2map

LOG_FILES = False

def preprocess_text(docs: Document) -> Dict:
    """
    Extract text from HTML and preprocess it using BeautifulSoup
    """
    t_flag1 = time.time()

    # Beautiful Soup Transformer
    bs_transformer = beautiful_soup_transformer.BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs,
        tags_to_extract=["p", "li", "div", "a", "span"],
        unwanted_tags=["script", "style", "noscript", "svg"],
    )
    # remove long white space
    docs_transformed = document_regex_sub(docs_transformed, r"\s+", " ")
    # remove unicode characters
    docs_transformed = document_regex_sub(docs_transformed, r"\\u[0-9A-Fa-f]{4}", "")

    t_flag2 = time.time()
    log.info(f"BeautifulSoupTransformer time: {t_flag2 - t_flag1}")

    if LOG_FILES:
        with open("src/log_data/docs_beautify.json", "w") as f:
            json.dump(document2map(docs_transformed), f)

    return docs_transformed


def docs_recursive_split(docs: Document, chunk_size: int = 400) -> List[Document]:
    t_flag1 = time.time()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=40
    )
    splits = splitter.split_documents(docs)

    t_flag2 = time.time()
    log.info(f"RecursiveCharacterTextSplitter time: {t_flag2 - t_flag1}")

    # convert to dictoinary
    splits = document2map(splits)

    if LOG_FILES:
        with open("src/log_data/splits.json", "w") as f:
            json.dump(splits, f)

    log.info(f"Total data splits: {len(splits)}")
    return splits


def contains_contacts(text: str) -> bool:
    """
    Check if the text contains email or phone number
    """
    # Regular expression patterns for emails and phone numbers
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    # phone_pattern = r"\b(?:\+\d{1,2}\s?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
    phone_pattern = r"\b(?:\+\d{1,3}\s?)?(?:\(\d{1,4}\)|\d{1,4})[\s.-]?\d{3,9}[\s.-]?\d{4}\b|\b\d{10}\b"

    contains_email = bool(re.search(email_pattern, text))
    contains_phone = bool(re.search(phone_pattern, text))

    return contains_email or contains_phone


def relevant_data(extracted_content):
    """
    Extract relevant data(checking for email and phone number) from the search results
    """
    t_flag1 = time.time()
    log.debug(f"before extraction: {len(extracted_content)}")
    data = [chunk for chunk in extracted_content if contains_contacts(chunk["content"])]
    log.debug(f"after extraction: {len(data)}")
    t_flag2 = time.time()
    log.info(f"Extraction time: {t_flag2 - t_flag1}")

    if LOG_FILES:
        with open("src/log_data/context_data.json", "w") as f:
            json.dump(data, f)

    return data


def process_data_docs(html_docs: Document, chunk_size: int = 400):
    """
    Process the data by extracting text from HTML, splitting it into chunks and extracting relevant data
    """
    docs = preprocess_text(docs=html_docs)

    data = docs_recursive_split(docs=docs, chunk_size=chunk_size)

    data = relevant_data(extracted_content=data)

    return data
