import logging
import os
from chromadb.config import Settings
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.embeddings import CohereEmbeddings


from constants import (
    DOCUMENT_MAP,
    INGEST_THREADS,
    LOCAL_EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
    COHERE_API_KEY,
)


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    file_extension = os.path.splitext(file_path)[1]
    loader_class = DOCUMENT_MAP.get(file_extension)
    if loader_class:
        loader = loader_class(file_path)
    else:
        raise ValueError("Document type is undefined")
    logging.debug(f"Loading file {file_path}")
    return loader.load()[0]


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory
    all_files = os.listdir(source_dir)
    paths = []
    for file_path in all_files:
        file_extension = os.path.splitext(file_path)[1]
        source_file_path = os.path.join(source_dir, file_path)
        if file_extension in DOCUMENT_MAP.keys():
            paths.append(source_file_path)

    # You can ingest the documents one at a time
    docs = []
    for file_path in paths:
        doc = load_single_document(file_path=file_path)
        docs.append(doc)

    # # Alternatively, you can load files in parallel (will be faster for multiple files)
    # # Have at least one worker and at most INGEST_THREADS workers
    # n_workers = min(INGEST_THREADS, max(len(paths), 1))
    # chunksize = round(len(paths) / n_workers)
    # docs = []
    # with ProcessPoolExecutor(n_workers) as executor:
    #     futures = []
    #     # split the load operations into chunks
    #     for i in range(0, len(paths), chunksize):
    #         # select a chunk of filenames
    #         filepaths = paths[i : (i + chunksize)]
    #         # submit the task
    #         future = executor.submit(load_document_batch, filepaths)
    #         futures.append(future)
    #     # process all results
    #     for future in as_completed(futures):
    #         # open the file and load the data
    #         contents, _ = future.result()
    #         docs.extend(contents)

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    # You can split different document types at different lengths.
    # For example, .py files might need a smaller chunk size
    text_docs, python_docs = [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, python_docs


def main():
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_documents, python_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)

    # Or you can do embedding locally if you've got a powerful computer. Even better, it's free!
    # EMBEDDING_MODEL_NAME = LOCAL_EMBEDDING_MODEL_NAME
    # embeddings = HuggingFaceInstructEmbeddings(
    #     model_name=EMBEDDING_MODEL_NAME,
    #     model_kwargs={"device": device_type},
    # )

    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY)

    db.persist()
    db = None


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
