# NOTE: This is for 'ingestion' of PDFs into Chroma. This is not for 'querying' Chroma.
#       Use this script to ingest PDFs into Chroma, and then use 2-convo.py to query Chroma w/ UI,
#       or use 1-chat.py to query Chroma from the command line, mostly for debugging.

import os
import json
import logging
import concurrent.futures
import re

# import torch
# from transformers import AutoTokenizer, AutoModel
import ebooklib
from ebooklib import epub
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.vectorstores.utils import filter_complex_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings


# List of files for verification artifacts, helpful for debugging when ingestion fails:
#
# cat started_list.txt| uniq | sort > started.txt
# cat processed_list.txt| uniq | sort > processed.txt
# diff started.txt processed.txt

processed_list_filename = "processed_list.txt"
started_list_filename = "started_list.txt"

# Download the required NLTK data
import nltk
nltk.download('averaged_perceptron_tagger')


# def extract_metadata_from_pdf(pdf_path):
#     reader = PdfReader(pdf_path)
#     metadata = reader.metadata
#     cleaned_metadata = {key.lstrip('/'): value for key, value in metadata.items()}
#     return cleaned_metadata


def remove_utf_surrogates(text):
    # Regex to match surrogate pairs
    surrogate_regex = re.compile(r"[\uD800-\uDFFF]")
    return surrogate_regex.sub("", text)


def extract_from_epub(epub_file):
    book = epub.read_epub(epub_file)
    text = ""

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            page_text = item.get_body_content().decode("utf-8")
            if page_text:
                # Remove surrogate pairs
                page_text = remove_utf_surrogates(page_text)

                # Normalize text encoding to handle special characters
                page_text = page_text.encode("utf-8", "replace").decode("utf-8") 
                text += page_text + "\n"

    metadata = {}
    metadata["file_path"] = epub_file if epub_file else None
    metadata["title"] = book.get_metadata("DC", "title")[0][0] if book.get_metadata("DC", "title") else None
    metadata["author"] = book.get_metadata("DC", "creator")[0][0] if book.get_metadata("DC", "creator") else None
    metadata["publisher"] = book.get_metadata("DC", "publisher")[0][0] if book.get_metadata("DC", "publisher") else None
    metadata["date"] = book.get_metadata("DC", "date")[0][0] if book.get_metadata("DC", "date") else None
    # metadata["description"] = book.get_metadata("DC", "description")[0][0] if book.get_metadata("DC", "description") else None
    metadata["language"] = book.get_metadata("DC", "language")[0][0] if book.get_metadata("DC", "language") else None
    metadata["rights"] = book.get_metadata("DC", "rights")[0][0] if book.get_metadata("DC", "rights") else None

    log.info(f"Metadata: {metadata}")
    return text, metadata

def extract_from_file(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            # Remove surrogate pairs
            page_text = remove_utf_surrogates(page_text)

            # Normalize text encoding to handle special characters
            page_text = page_text.encode("utf-8", "replace").decode("utf-8")
            text += page_text + "\n"

    # Convert metadata to a regular dictionary and ensure all values are simple data types
    metadata = {}
    metadata["file_path"] = pdf_file if pdf_file else None
    if reader.metadata is not None:
        for key, value in reader.metadata.items():
            key = key.lstrip("/")
            # Convert value to string to ensure compatibility with Chroma
            metadata[key] = str(value) if value is not None else None

    return text, metadata


def process_file(input_file):
    log.info(f"Started {input_file}...")

    # Add the file name to started_list_filename
    with open(started_list_filename, "a") as file:
        file.write(input_file + "\n")

    all_docs = []
    if input_file.endswith(".pdf"):
        pdf_file = os.path.join(input_file_path, input_file)
        text, metadata = extract_from_file(pdf_file)

        docs = text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )

        for doc in docs:
            all_docs.append(doc)

        if all_docs:
            # print(all_docs[0].page_content)
            # Add the file name to processed_list_filename
            with open(processed_list_filename, "a") as file:
                file.write(input_file + "\n")

            log.info(f"Finished with {pdf_file}.")
            return filter_complex_metadata(all_docs)
        else:
            log.info(f"No documents found in {pdf_file}.")
            return None

    elif input_file.endswith(".epub"):
        epub_file = os.path.join(input_file_path, input_file)
        text, metadata = extract_from_epub(epub_file)
        docs = text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata],
        )

        for doc in docs:
            all_docs.append(doc)

        if all_docs:
            # print(all_docs[0].page_content)
            # Add the file name to processed_list_filename
            with open(processed_list_filename, "a") as file:
                file.write(input_file + "\n")

            log.info(f"Finished with {epub_file}.")
            return filter_complex_metadata(all_docs)
        else:
            log.info(f"No documents found in {epub_file}.")
            return None


## Begin processing PDFs

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1200,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)


# Initialize the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(
    model_name="all-mpnet-base-v2"
)  # all-MiniLM-L6-v2


# Directory containing PDF/epub files
#
# BTW, not all PDFs and EPUBs work w/ these readers...  I had 1 PDF and 3 EPUBs that didn't work so far
# out of about 150+ files
input_file_path = "../input"

# Define the number of thread workers
num_workers = 7

# Process PDFs in parallel
all_docs = []
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    future_to_pdf = {
        executor.submit(process_file, input_file): input_file
        for input_file in os.listdir(input_file_path)
    }

    for future in concurrent.futures.as_completed(future_to_pdf):
        docs = future.result()

        if docs:
            all_docs.extend(docs)


log.info(f"Finished processing all PDFs. Got {len(all_docs)} documents.")
if len(all_docs) == 0:
    log.info("No documents found. Exiting...")
    exit()

# Initialize or create a new Chroma DB with all documents
chroma = Chroma.from_documents(
    documents=all_docs,
    persist_directory="./test/chroma-all-mpnet-base-v2",
    embedding=embedding_function,
)
chroma.persist()


# # Save the mapping to a file
# doc_info_mapping = {i: doc.metadata['file_path'] for i, doc in enumerate(all_docs)}
# with open("doc_info_mapping.json", "w") as file:
#     json.dump(doc_info_mapping, file)
