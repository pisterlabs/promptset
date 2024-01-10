#!/usr/bin/env python3

# Imports
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from programengineergpt.utils.colors import Color


class Retriever:
    def __init__(self, col_name=None):
        if col_name is None:
            Color.print("\n{Y}Please enter collection name for code base: ")
            self.collection_name = col_name = input("Collection Name: ")

        self.collection_name = col_name
        self.embedding_function = OpenAIEmbeddingFunction()
        self.chromadb_client = chromadb.Client()
        self.collection = self.chromadb_client.get_collection(
            name=self.collection_name, embedding_function=self.embedding_function
        )

        Color.print("\n\n{G}Retrieving Code Index...\n")

    def retrive_code_index(self):
        return self.collection
