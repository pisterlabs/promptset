import glob
import os
from multiprocessing import Pool
from typing import List, Optional
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm

from scripts.app_environment import (
    ingest_chunk_size,
    ingest_chunk_overlap,
    ingest_embeddings_model,
    ingest_persist_directory,
    ingest_source_directory,
    args,
    chromaDB_manager,
    gpu_is_enabled)
from scripts.app_utils import LOADER_MAPPING, load_single_document


def load_documents(source_dir: str, collection_name: Optional[str], ignored_files: List[str] = []) -> List[Document]:
    collection_dir = os.path.join(source_dir, collection_name) if collection_name else source_dir
    print(f"Loading documents from {collection_dir}")
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(collection_dir, f"*{ext}"), recursive=False)
        )
    filtered_files = [file_path for file_path in all_files if os.path.isfile(file_path) and file_path not in ignored_files]

    with Pool(processes=min(8, os.cpu_count())) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                if isinstance(docs, dict):
                    print(" - " + docs['file'] + ": error: " + str(docs['exception']))
                    continue

                print(f"\n\033[32m\033[2m\033[38;2;0;128;0m{docs[0].metadata.get('source', '')} \033[0m")
                results.extend(docs)
                pbar.update()

    return results


def process_documents(collection_name: Optional[str] = None, ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split them into chunks.
    """
    # db_name = args.ingest_dbname or os.path.basename(source_directory)
    documents = load_documents(source_directory, collection_name if db_name != collection_name else None, ignored_files)

    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=ingest_chunk_size if ingest_chunk_size else args.ingest_chunk_size,
        chunk_overlap=ingest_chunk_overlap if ingest_chunk_overlap else args.ingest_chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text (max. {ingest_chunk_size} tokens each)")
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if a Chroma vectorstore already exists in the given directory.
    :param persist_directory: The path of the vectorstore directory.
    :return: True if the vectorstore exists, False otherwise.
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


def prompt_user():
    print(f"Using current ingest_source_directory: {ingest_source_directory}")

    return ingest_source_directory, ingest_persist_directory


def create_embeddings():
    embeddings_kwargs = {'device': 'cuda'} if gpu_is_enabled else {}
    return HuggingFaceEmbeddings(
        model_name=ingest_embeddings_model if ingest_embeddings_model else args.ingest_embeddings_model,
        model_kwargs=embeddings_kwargs
    )


def get_chroma(collection_name: str, embeddings, persist_dir):
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=chromaDB_manager.get_chroma_setting(persist_dir)
    )


def process_and_add_documents(collection, chroma_db, collection_name):
    ignored_files = [metadata['source'] for metadata in collection['metadatas']]
    texts = process_documents(collection_name=collection_name, ignored_files=ignored_files)
    num_elements = len(texts)
    index_metadata = {"elements": num_elements}
    print("Creating embeddings. May take some minutes...")
    chroma_db.add_documents(texts, index_metadata=index_metadata)


def process_and_persist_db(database, collection_name):
    print(f"Collection: {collection_name}")
    process_and_add_documents(database.get(), database, collection_name)
    database.persist()


def create_and_persist_db(embeddings, texts, persist_dir, collection_name):
    num_elements = len(texts)
    index_metadata = {"elements": num_elements}
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
        client_settings=chromaDB_manager.get_chroma_setting(persist_dir),
        index_metadata=index_metadata
    )
    db.persist()


def main(source_dir: str, persist_dir: str, db_name: str, sub_collection_name: Optional[str] = None):
    embeddings = create_embeddings()
    collection_name = sub_collection_name or db_name

    if does_vectorstore_exist(persist_dir):
        print(f"Appending to existing vectorstore at {persist_dir}")
        db = get_chroma(collection_name, embeddings, persist_dir)
        process_and_persist_db(db, collection_name)
    else:
        print(f"Creating new vectorstore from {source_dir}")
        texts = process_documents(collection_name=collection_name, ignored_files=[])
        create_and_persist_db(embeddings, texts, persist_dir, collection_name)

    print("Ingestion complete!")


source_directory, persist_directory = prompt_user()
db_name = os.path.basename(persist_directory)


def ingest():
    documents = load_documents(source_directory, db_name)
    if documents:
        main(source_directory, persist_directory, db_name)


if __name__ == "__main__":
    # ingest()
    main(source_directory, persist_directory, db_name)
