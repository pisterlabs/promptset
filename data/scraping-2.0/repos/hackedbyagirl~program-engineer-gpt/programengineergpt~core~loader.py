#!/usr/bin/env python3

import os
import chromadb
import fnmatch

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from programengineergpt.utils.colors import Color
from programengineergpt.utils.file_processing import extensions_include, extensions_ignore, dirs_ignore


class CodeLoader:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddingFunction()
        self.chromadb_client = chromadb.Client()

        self.code_files = []
        self.files = []
        self.chunks = []
        self.extensions = extensions_include
        self.extensions_ignore = extensions_ignore
        self.directories_ignore = dirs_ignore
            
        Color.print("\n\n{G}Launching Code Loader...\n")

    def load_online_repo(self, url):
        """
        Load all code files from the specified repository.
        """
        if "github.com/" in url or "gitlab.com/" in url:
            # Clone repository
            try:
                Color.print("{G}Step 1: {W}Retrieving Code from Online Repository")
                os.system(f"git clone --quiet {url} temp_repo")
                vector_store = self.load("temp_repo")

            except Exception as e:
                Color.print(
                    "{R}!!!: {W}Failed to clone GitHub repository from {Y}" + url
                )
                Color.p_error(e)

        return vector_store

    def load_directory(self, path):
        """
        Load all code files from a specified directory.
        """
        # Get code from provided directory
        Color.print("{G}Step 1: {W}Retrieving Code from Local Repository")
        if not os.path.isdir(path):
            raise Exception(f"Invalid local directory: {path}")

        vector_store = self.load(path)
        return vector_store

    def load_current_directory(self):
        """
        Load all code files from the current directory.
        """
        Color.print("{G}Step 1: {W}Retrieving Code from Current Directory")
        vector_store = self.load(os.getcwd())
        return vector_store

    def load(self, root_dir, col_name=None):
        """
        Load all code files from the root directory.
        """
        try:
            Color.print("{G}Step 2: {W}Loading Code for Indexing and Embedding")
            for dirpath, dirnames, filenames in os.walk(root_dir):
                # Modifying dirnames list will update os.walk behavior
                dirnames[:] = [d for d in dirnames if d not in self.directories_ignore]
                for file in filenames:
                    # Check if file matches any ignore pattern and skip loading
                    if any(fnmatch.fnmatch(file, pattern) for pattern in self.extensions_ignore):
                        continue  # Skip this file
                        
                    if file.split(".")[-1] in self.extensions:
                        with open(os.path.join(dirpath, file), 'r') as f:
                            self.files.append((file, f.read()))

        except Exception as e:
            Color.print("{R}!!!: {W}Failed to load code files from {Y}" + root_dir)
            Color.p_error(e)

        # print loaded files
        len_files = str(len(self.files))
        Color.print("{Y}Number of Loaded Files: {W}" + len_files)

        # Split code
        Color.print("{G}Step 3: {W}Splitting and Chunking Files")
        self.chunks = self.split_code()

        # embed code
        Color.print("{G}Step 4: {W}Embedding and Uploading to ChromaDB")

        if col_name is None:
            Color.print("\n{Y}Please enter a Collection Name for you Code Base: ")
            col_name = input("Collection Name: ")

        code_collection = self.chromadb_client.create_collection(
            name=col_name, embedding_function=self.embedding_function
        )

        counter = 0
        batch_size = 100
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            documents = []
            ids = []
            metadatas = []
            for full_path, chunks in batch:
                for j, chunk in enumerate(chunks):
                    documents.append(chunk)
                    # Use the counter to create unique IDs
                    ids.append(f"{full_path}_{counter}")
                    metadatas.append({"filename": full_path})
                    # Increment the counter
                    counter += 1
            code_collection.add(ids=ids, documents=documents, metadatas=metadatas)

        if os.path.exists("temp_repo"):
            Color.print('{G}Step 5: {W}Removing "temp_repo" contents')
            os.system("rm -rf temp_repo")

        return code_collection

    def split_code(self):
        """Split the code of all files into chunks."""
        all_chunks = []
        for filename, code in self.files:
            chunks = [code[i : i + 1000] for i in range(0, len(code), 1000)]
            all_chunks.append((filename, chunks))
        return all_chunks

    def remove_temp_dir(self):
        """
        Remove 'temp_repo' if a repository was cloned
        """
        if os.path.exists("temp_repo"):
            Color.print('{G}Step 5: {W}Removing "temp_repo" contents')
            os.system("rm -rf temp_repo")
