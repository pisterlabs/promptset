#!/usr/bin/env python3
import glob
import os
import re
import sys
from multiprocessing import Pool
from time import monotonic
from typing import List, Optional

from dotenv import set_key
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
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
from scripts.app_utils import display_directories, LOADER_MAPPING, load_single_document


def load_documents(source_dir: str, collection_name: Optional[str], ignored_files=None) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files.
    :param source_dir: The path of the source documents directory.
    :param collection_name: The name of the collection to exclude files from.
    :param ignored_files: A list of filenames to be ignored.
    :return: A list of Document objects loaded from the source documents.
    """
    if ignored_files is None:
        ignored_files = []
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
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")

    texts = RecursiveCharacterTextSplitter(
        chunk_size=ingest_chunk_size if ingest_chunk_size else args.ingest_chunk_size,
        chunk_overlap=ingest_chunk_overlap if ingest_chunk_overlap else args.ingest_chunk_overlap
    ).split_documents(documents)

    print(f"Split into {len(texts)} chunks of text (max. {ingest_chunk_size} tokens each)")
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if the required structure exists in the given directory.

    The structure is defined as:
    - A directory in UUID format.
    - Inside the UUID directory: some .bin files and a .pickle file.
    - A .sqlite3 file in the persisted directory.

    :param persist_directory: The path of the directory to check.
    :return: True if the structure exists, False otherwise.
    """

    # Check if there's a .sqlite3 file in the persist_directory
    if not glob.glob(os.path.join(persist_directory, '*.sqlite3')):
        return False

    # Check for UUID formatted directories
    uuid_pattern = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')
    uuid_directories = [d for d in os.listdir(persist_directory) if os.path.isdir(os.path.join(persist_directory, d)) and uuid_pattern.match(d)]

    for uuid_dir in uuid_directories:
        bin_files = glob.glob(os.path.join(persist_directory, uuid_dir, '*.bin'))
        pickle_files = glob.glob(os.path.join(persist_directory, uuid_dir, '*.pickle'))

        if bin_files and pickle_files:
            return True

    return False


def prompt_user():
    """
    Prompts the user to select an existing directory or create a new one to store source material.
    If an existing directory is selected, it checks if the directory is empty and prompts the user to create files
    in the directory if it is empty. It sets the directory paths as environment variables and returns them.
    :return: The selected source directory path, the selected database directory path, and the collection name.
    """

    def _create_directory(directory_name):
        """
        Creates a new directory with the given directory_name in the ./source_documents directory.
        It also creates a corresponding directory in the ./db directory for the database files.
        It sets the directory paths as environment variables and returns them.
        :param directory_name: The name for the new directory.
        :return: The path of the new directory and the path of the database directory.
        """
        directory_path = os.path.join(".", "source_documents", directory_name)
        db_path = os.path.join(".", "db", directory_name)
        os.makedirs(directory_path)
        os.makedirs(db_path)
        set_key('.env', 'INGEST_SOURCE_DIRECTORY', directory_path)
        set_key('.env', 'INGEST_PERSIST_DIRECTORY', db_path)
        print(f"Created new directory: {directory_path}")
        return directory_path, db_path

    while True:
        print(f"\033[94mSelect an option or 'q' to quit:\n\033[0m")
        print("1. Select an existing directory")
        print("2. Create a new directory")
        print(f"3. Use current ingest_source_directory: {ingest_source_directory}")

        user_choice = input('\nEnter your choice ("q" for quit): ').strip()

        if user_choice == "1":
            directories = display_directories()
            while True:  # Keep asking until we get a valid directory number
                existing_directory = input("\n\033[94mEnter the number of the existing directory (q for quit, b for back): \033[0m")
                if existing_directory == 'q':
                    raise SystemExit
                elif existing_directory == 'b':
                    break
                try:
                    selected_directory = directories[int(existing_directory) - 1]
                    selected_directory_path = os.path.join(".", "source_documents", selected_directory)
                    selected_db_path = os.path.join(".", "db", selected_directory)
                    if not os.listdir(selected_directory_path):
                        print(f"\033[91m\033[1m[!]\033[0m Selected directory: '{selected_directory}' is empty \033[91m\033[1m[!]\033[0m")
                        directories = display_directories()  # Display directories again if the selected one is empty
                    else:
                        if not os.path.exists(selected_db_path):
                            os.makedirs(selected_db_path)
                        set_key('.env', 'INGEST_SOURCE_DIRECTORY', selected_directory_path)
                        set_key('.env', 'INGEST_PERSIST_DIRECTORY', selected_db_path)
                        print(f"Selected directory: {selected_directory_path}")
                        return selected_directory_path, selected_db_path
                except (ValueError, IndexError):
                    print("\n\033[91m\033[1m[!] \033[0mInvalid choice. Please try again.\033[91m\033[1m[!] \033[0m\n")
                    directories = display_directories()  # Display directories again if the input is invalid
        elif user_choice == "2":
            new_directory_name = input("Enter the name for the new directory: ")
            selected_directory_path, selected_db_path = _create_directory(new_directory_name)
            input("Place your source material into the new folder and press enter to continue...")
            return selected_directory_path, selected_db_path
        elif user_choice == "3":
            return ingest_source_directory, ingest_persist_directory
        elif user_choice == "q":
            exit(0)
        else:
            print("\n\033[91m\033[1m[!] \033[0mInvalid choice. Please try again.\033[91m\033[1m[!] \033[0m\n")


def create_embeddings():
    embeddings_kwargs = {'device': 'cuda'} if gpu_is_enabled else {'device': 'cpu'}
    return HuggingFaceInstructEmbeddings(
        model_name=ingest_embeddings_model if ingest_embeddings_model else args.ingest_embeddings_model,
        model_kwargs=embeddings_kwargs
    )


def get_chroma(collection_name: str, embeddings, persist_dir):
    return Chroma(
        persist_directory=persist_dir,
        collection_name=collection_name,
        embedding_function=embeddings,
        client_settings=chromaDB_manager.get_chroma_setting(persist_dir),
        client=chromaDB_manager.get_client(collection_name),
    )


def process_and_add_documents(collection, chroma_db, collection_name):
    ignored_files = [metadata['source'] for metadata in collection['metadatas']]
    texts = process_documents(collection_name=collection_name, ignored_files=ignored_files)
    num_elements = len(texts)
    collection_metadata = {"elements": num_elements}
    print(f"Creating embeddings. May take some minutes...")
    chroma_db.add_documents(texts, collection_metadata=collection_metadata)


def process_and_persist_db(database, collection_name):
    print(f"Collection: {collection_name}")
    process_and_add_documents(database.get(), database, collection_name)


def create_and_persist_db(embeddings, texts, persist_dir, collection_name):
    num_elements = len(texts)
    collection_metadata = {"elements": num_elements}
    Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
        client_settings=chromaDB_manager.get_chroma_setting(persist_dir),
        client=chromaDB_manager.get_client(collection_name),
        collection_metadata=collection_metadata
    )


def main(source_dir: str, persist_dir: str, db_name: str, sub_collection_name: Optional[str] = None):
    embeddings = create_embeddings()
    collection_name = sub_collection_name or db_name

    start_time = monotonic()

    if does_vectorstore_exist(persist_dir):
        print(f"Appending to existing vectorstore at {persist_dir}")
        db = get_chroma(collection_name, embeddings, persist_dir)
        process_and_persist_db(db, collection_name)
    else:
        print(f"Creating new vectorstore from {source_dir}")
        texts = process_documents(collection_name=collection_name, ignored_files=[])
        create_and_persist_db(embeddings, texts, persist_dir, collection_name)

    print("Ingestion complete! You can now run scrapalot_main.py to query your documents")
    print(f"\033[94mTook {round(((monotonic() - start_time) / 60), 2)} min to process the ingestion!\033[0m")


if __name__ == "__main__":
    try:

        if args.ingest_all:  # Check if --ingest-all is provided
            base_dir = os.path.join(".", "source_documents")
            sorted_list = sorted(os.listdir(base_dir))
            for dir_name in sorted_list:  # Iterate over all directories in source_documents
                if not dir_name.startswith("."):
                    db_name = dir_name
                    source_directory = os.path.join(".", "source_documents", db_name)
                    persist_directory = os.path.join(".", "db", db_name)

                    if not os.path.exists(source_directory):
                        os.makedirs(source_directory)

                    if not os.path.exists(persist_directory):
                        os.makedirs(persist_directory)

                    if args.collection:
                        sub_collection_name = args.collection
                        main(source_directory, persist_directory, db_name, sub_collection_name)
                    else:
                        main(source_directory, persist_directory, db_name)
        elif args.ingest_dbname:
            db_name = args.ingest_dbname
            source_directory = os.path.join(".", "source_documents", db_name)
            persist_directory = os.path.join(".", "db", db_name)

            if not os.path.exists(source_directory):
                os.makedirs(source_directory)

            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)

            if args.collection:
                sub_collection_name = args.collection
                main(source_directory, persist_directory, db_name, sub_collection_name)
            else:
                main(source_directory, persist_directory, db_name)
        else:
            source_directory, persist_directory = prompt_user()
            db_name = os.path.basename(persist_directory)
            main(source_directory, persist_directory, db_name)
    except SystemExit:
        print("\n\033[91m\033[1m[!] \033[0mExiting program! \033[91m\033[1m[!] \033[0m")
        sys.exit(1)
