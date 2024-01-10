#   Copyright [2023] [Sunholo ApS]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import logging
import pathlib
import tempfile
import os
import re
import json

from google.cloud import storage
from langchain.schema import Document

from .gcs import add_file_to_gcs
from .splitter import chunk_doc_to_docs
from .pdfs import split_pdf_to_pages
from .publish import publish_if_urls
from . import loaders
from ..utils.parsers import extract_urls

def handle_gcs_message(message_data: str, metadata: dict, vector_name: str):
    # Process message from Google Cloud Storage
    logging.debug("Detected gs://")
    bucket_name, file_name = message_data[5:].split("/", 1)

    # Create a client
    storage_client = storage.Client()

    # Download the file from GCS
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    file_name=pathlib.Path(file_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_file_path = os.path.join(temp_dir, file_name.name)
        blob.download_to_filename(tmp_file_path)

        if file_name.suffix == ".pdf":
            pages = split_pdf_to_pages(tmp_file_path, temp_dir)
            if len(pages) > 1: # we send it back to GCS to parrallise the imports
                logging.info(f"Got back {len(pages)} pages for file {tmp_file_path}")
                for pp in pages:
                    gs_file = add_file_to_gcs(pp, vector_name=vector_name, bucket_name=bucket_name, metadata=metadata)
                    logging.info(f"{gs_file} is now in bucket {bucket_name}")
                logging.info(f"Sent split pages for {file_name.name} back to GCS to parrallise the imports")
                return None
        else:
            # just original temp file
            pages = [tmp_file_path]

        the_metadata = {
            "source": message_data,
            "type": "file_load_gcs",
            "bucket_name": bucket_name
        }
        metadata.update(the_metadata)

        docs = []
        for page in pages:
            logging.info(f"Sending file {page} to loaders.read_file_to_document {metadata}")
            docs2 = loaders.read_file_to_document(page, metadata=metadata)
            docs.extend(docs2)

        chunks = chunk_doc_to_docs(docs, file_name.suffix)

        return chunks, metadata

def handle_google_drive_message(message_data: str, metadata: dict):
    # Process message from Google Drive
    logging.info("Got google drive URL")
    urls = extract_urls(message_data)

    docs = []
    for url in urls:
        metadata["source"] = url
        metadata["url"] = url
        metadata["type"] = "url_load"
        doc = loaders.read_gdrive_to_document(url, metadata=metadata)
        if doc is None:
            logging.info("Could not load any Google Drive docs")
        else:
            docs.extend(doc)

    chunks = chunk_doc_to_docs(docs)

    return chunks, metadata

def handle_github_message(message_data: str, metadata: dict):
    # Process message from GitHub
    logging.info("Got GitHub URL")
    urls = extract_urls(message_data)

    branch="main"
    if "branch:" in message_data:
        match = re.search(r'branch:(\w+)', message_data)
        if match:
            branch = match.group(1)
    
    logging.info(f"Using branch: {branch}")

    docs = []
    for url in urls:
        metadata["source"] = url
        metadata["url"] = url
        metadata["type"] = "url_load"
        doc = loaders.read_git_repo(url, branch=branch, metadata=metadata)
        if doc is None:
            logging.info("Could not load GitHub files")
        else:
            docs.extend(doc)
    
    chunks = chunk_doc_to_docs(docs)
    
    return chunks, metadata

def handle_http_message(message_data: str, metadata: dict):
    # Process message from a generic HTTP URL
    logging.info(f"Got http message: {message_data}")

    # just in case, extract the URL again
    urls = extract_urls(message_data)

    docs = []
    for url in urls:
        metadata["source"] = url
        metadata["url"] = url
        metadata["type"] = "url_load"
        doc = loaders.read_url_to_document(url, metadata=metadata)
        docs.extend(doc)

    chunks = chunk_doc_to_docs(docs)

    return chunks, metadata

def handle_json_content_message(message_data: str, metadata: dict, vector_name: str):
    logging.info("No tailored message_data detected, processing message json")
    # Process message containing direct JSON content
    the_json = json.loads(message_data)
    the_metadata = the_json.get("metadata", {})
    metadata.update(the_metadata)
    the_content = the_json.get("page_content", None)

    if metadata.get("source", None) is not None:
        metadata["source"] = "No source embedded"

    if the_content is None:
        logging.info("No content found")
        return {"metadata": "No content found"}
    
    docs = [Document(page_content=the_content, metadata=metadata)]

    publish_if_urls(the_content, vector_name)

    chunks = chunk_doc_to_docs(docs)

    return chunks, metadata