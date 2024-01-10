import os
import tiktoken
from tqdm.auto import tqdm
from typing import List
from pinecone import Index
from langchain_core.documents import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from tkinter import filedialog

from utils import update_existing_sources, LOCAL_SOURCES_FILEPATH

EXTENSIONS = [".pdf", ".docx", ".pptx"]
ENCODING_NAME = "cl100k_base"
MAX_CHUNK_SIZE = 400
UPSERT_BATCH_LIMIT = 100


def tiktoken_len(text):
    """Returns the number of tokens in a text."""
    tokenizer = tiktoken.get_encoding(ENCODING_NAME)

    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )

    return len(tokens)


def parse_single_document(path: str):
    """Parses a single document and returns a list of Document objects."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )

    loader = None

    if path.endswith(".docx"):
        loader = Docx2txtLoader(path)
    elif path.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(path)
    elif path.endswith(".pdf"):
        loader = PyPDFLoader(path)

    return loader.load_and_split(text_splitter)


def upsert_documents(embed: OpenAIEmbeddings, index: Index, texts: list, metadatas: list):
    """Upserts a batch of documents to the given index."""
    print(f"Upserting {len(texts)} documents.")

    ids = [str(uuid4()) for _ in range(len(texts))]
    embeds = embed.embed_documents(texts)
    index.upsert(vectors=zip(ids, embeds, metadatas))


def get_title_from_filepath(filepath: str):
    return os.path.basename(filepath)


def process_documents_for_upsert(embed: OpenAIEmbeddings, index: Index, documents: List[Document]):
    """Processes uploaded docuemnts and calls upsert_documents() in batches."""
    texts = []
    metadatas = []

    for _, record in enumerate(tqdm(documents)):
        metadata = {
            'page': str(record.metadata['page']) if "page" in record.metadata else "N/A",
            'source': record.metadata['source'],
            'title': get_title_from_filepath(record.metadata['source']),
            'text': record.page_content,
        }

        texts.append(record.page_content)
        metadatas.append(metadata)

        if len(texts) >= UPSERT_BATCH_LIMIT:
            upsert_documents(embed, index, texts, metadatas)

            texts = []
            metadatas = []

    if len(texts) > 0:
        upsert_documents(embed, index, texts, metadatas)


def retrieve_files_from_folderpath(folderpath: str, extensions: list[str]):
    """Returns list of paths of all files with valid extensions."""
    file_list = []

    for root, _, files in os.walk(folderpath):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_list.append(os.path.join(root, file))

    return file_list


def crawl_and_upsert(embed: OpenAIEmbeddings, index: Index, previously_upserted: bool, existing_sources: list[str]):
    """Crawls a user-selected folder and upserts all files with valid extensions."""
    filepaths = []

    print()
    # NOTE: on older versions of tkinter, repeated calls will result in segfault.
    # https://github.com/python/cpython/issues/92603
    while (True):
        if previously_upserted:
            print(
                "Note that older versions of tkinter will error if you open up the select dialog multiple times. ", end="")
        crawling_option = input(
            "Would you like to (1) select individual files or (2) select a folder? ")
        if crawling_option == "1":
            filepaths = filedialog.askopenfilenames(
                filetypes=[(f"{ext} files", f"*{ext}") for ext in EXTENSIONS])
            break
        elif crawling_option == "2":
            folderpath = filedialog.askdirectory()
            filepaths = retrieve_files_from_folderpath(folderpath, EXTENSIONS)
            break
        else:
            print("Please enter either 1 or 2")

    previously_upserted = True

    # Warn the user if uploading many files to avoid accidental uploads
    if len(filepaths) > 10:
        continue_input = input(
            f"There are {len(filepaths)} files with extensions {EXTENSIONS} in this folder. Are you sure you want to continue? (y/N): ")
        if continue_input != "y":
            print("Aborting!")
            return

    upserted_filepaths = []

    # iterate through all filepaths
    for filepath in filepaths:
        # TODO: store date uploaded and compare to metadata to see if updated since last upload
        if filepath in existing_sources:
            print(
                f"Skipping {get_title_from_filepath(filepath)} because it was already parsed.")
        else:
            # TODO: only call this once on entire set of documents
            parsed_document = parse_single_document(filepath)
            process_documents_for_upsert(embed, index, parsed_document)
            upserted_filepaths.append(filepath)

    with open(LOCAL_SOURCES_FILEPATH, 'a') as file:
        for filepath in upserted_filepaths:
            file.write(filepath + "\n")

    update_existing_sources(existing_sources)
