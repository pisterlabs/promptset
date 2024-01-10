#!/usr/bin/env python3
import os
import glob
from datetime import datetime
from functools import reduce
from itertools import chain
import re
from typing import List, Generator
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm
import logging as log
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader, PDFMinerLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.docstore.document import Document

from privateGPT.msg_loader import MsgLoader

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

from .global_vars import CHROMA_SETTINGS
import chromadb
from chromadb.api.segment import API

# Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 4300
chunk_overlap = 0
pattern = r"QUOTE DATE:\nQUOTE NO:\n\d{2}/\d{2}/\d{4}\n\d+\n"


# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".pdf2": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".msg": (MsgLoader,  {})
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")

def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True)
        )
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    results = map(load_single_document,filtered_files)
    results = list(results)[0]
    # results = reduce(lambda acc, item: acc + item, map(chain.from_iterable, results))

    # with Pool(processes=os.cpu_count()) as pool:
    #     results = []
    #     with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
    #         for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
    #             results.extend(docs)
    #             pbar.update()

    return results

def reptext():
    date = datetime.now().strftime('%m/%d/%Y')
    quote_no = "4"

    # Replacement string with example values
    replacement = f"QUOTE DATE:\nQUOTE NO:\n{date}\n{quote_no}\n"
    return replacement

def process_documents(ignored_files: List[str] = [], source_folder=source_directory) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_folder}")
    documents = load_documents(source_folder, ignored_files)
    documents = list(documents)
    if not documents:
        print("No new documents to load")
        raise(Exception('No New Docs to load'))
    print(f"Loaded {len(documents)} new documents from {source_folder}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # for doc in documents[1:]:
    #     if 'Carahsoft' in doc.metadata['source']:
    #         doc.page_content = re.sub(pattern, reptext(), doc.page_content)
    documents = text_splitter.split_documents(documents)
    print(f"Split into {len(documents)} chunks of text (max. {chunk_size} tokens each)")
    return documents


def process_text_documents(text:str, metadata=None) -> List[Document]:
    """
    Load documents and split in chunks
    """
    if metadata is None:
        metadata = {}
    print(f"Loading documents from text: {text}")
    documents =  [Document(page_content=text, metadata=metadata) if metadata else Document(page_content=text) ]
    if not documents:
        print("No new documents to load")
        raise Exception('Doc is empty error:'+text)
    print(f"Loaded {len(documents)} new documents from text")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents)
    print(f"Split into {len(documents)} chunks of text (max. {chunk_size} tokens each)")
    return documents
def batch_chromadb_insertions(chroma_client: API, documents: List[Document]) -> Generator[List[Document], None, None]:
    """
    Split the total documents to be inserted into batches of documents that the local chroma client can process
    """
    # Get max batch size.
    max_batch_size = chroma_client.max_batch_size
    for i in range(0, len(documents), max_batch_size):
        yield documents[i:i + max_batch_size]


def does_vectorstore_exist(persist_directory: str, embeddings: HuggingFaceEmbeddings) -> bool:
    """
    Checks if vectorstore exists
    """
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    if not db.get()['documents']:
        return False
    return True



def text_main(persist_directory:str, extracted_text:str, metadata:str, openai=False):
    # Create embeddings
    from langchain.embeddings import OpenAIEmbeddings

    if openai:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    else:
        openai.api_base = os.environ['OPENAI_API_BASE']
        openai.api_version = os.environ['OPENAI_API_VERSION']
        openai.api_key = os.environ['OPENAI_API_KEY']
        persist_directory = os.environ.get('PERSIST_DIRECTORY')
        embeddings = OpenAIEmbeddings(deployment=embeddings_model_name)

    # Chroma client
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)

    if does_vectorstore_exist(persist_directory, embeddings):
        # Update and store locally vectorstore
        log.info(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
        collection = db.get()
        documents = process_text_documents(extracted_text,metadata)
        log.info(f"Creating embeddings. May take some minutes...")
        for batched_chromadb_insertion in batch_chromadb_insertions(chroma_client, documents):
            db.add_documents(batched_chromadb_insertion)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        documents = process_text_documents(extracted_text,metadata)
        print(f"Creating embeddings. May take some minutes...")
        # Create the db with the first batch of documents to insert
        batched_chromadb_insertions = batch_chromadb_insertions(chroma_client, documents)
        first_insertion = next(batched_chromadb_insertions)
        db = Chroma.from_documents(first_insertion, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)
        # Add the rest of batches of documents
        for batched_chromadb_insertion in batched_chromadb_insertions:
            db.add_documents(batched_chromadb_insertion)

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")

def main(direct:str=None):
    # Create embeddings
    global persist_directory
    if direct:
        persist_directory = direct
    if '-ada-' in embeddings_model_name:
        embeddings = OpenAIEmbeddings(deployment=embeddings_model_name,chunk_size=1, engine=embeddings_model_name)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # Chroma client
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)

    if does_vectorstore_exist(persist_directory, embeddings):
        # Update and store locally vectorstore
        log.info(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
        collection = db.get()
        documents = process_documents([metadata['source'] for metadata in collection['metadatas']], source_folder=persist_directory)
        log.info(f"Creating embeddings. May take some minutes...")
        for batched_chromadb_insertion in batch_chromadb_insertions(chroma_client, documents):
            db.add_documents(batched_chromadb_insertion)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        documents = process_documents(source_folder=persist_directory)
        print(f"Creating embeddings. May take some minutes...")
        # Create the db with the first batch of documents to insert
        batched_chromadb_insertions = batch_chromadb_insertions(chroma_client, documents)
        first_insertion = next(batched_chromadb_insertions)
        db = Chroma.from_documents(first_insertion, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client)
        # Add the rest of batches of documents
        for batched_chromadb_insertion in batched_chromadb_insertions:
            db.add_documents(batched_chromadb_insertion)

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()


