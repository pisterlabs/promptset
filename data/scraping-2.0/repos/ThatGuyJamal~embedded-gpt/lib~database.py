from typing import Any, Optional, List

import chromadb
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OllamaEmbeddings

class Database:
    """
    Database class for storing embeddings and querying them
    """

    def __init__(self) -> None:
        self.chroma_client = chromadb.PersistentClient("./db/embedded-gpt")
        self._collection_name = "embedded-gpt"
        self.ollama_embed_func = OllamaEmbeddings(base_url='http://localhost:11434', model="codellama:7b")

        self.langchain_chroma = Chroma(
            client=self.chroma_client,
            embedding_function=self.ollama_embed_func,
            collection_name="embedded-gpt",
        )

    def add_embeds(self, docs: List[chromadb.Documents], meta: Optional[Any] = None):
        """
        Add embeddings to the database
        """
        print("Adding %d documents to database and collection %s" % (len(docs), self._collection_name))
        self.langchain_chroma.from_documents(collection_name=self._collection_name, documents=docs, embedding=self.ollama_embed_func, collection_metadata=meta, client=self.chroma_client)
        print("Done adding documents to database")

    def delete_collection(self, collection_name: str):
        """
        Delete a collection from the database
        """
        if collection_name == "":
            print("No collection name provided for deletion, try again.")
            return

        valid_collections = self.get_collection_names(as_list=True)
        if collection_name not in valid_collections:
            print("Collection %s does not exist, try again." % collection_name)
            return

        self.langchain_chroma.delete_collection(collection_name)
        print("Deleted collection %s" % collection_name)

    def total_documents(self) -> int:
        """
        Get the total number of documents in the database
        """

        if len(self.chroma_client.list_collections()) == 0:
            return 0

        return self.langchain_chroma._collection.count()

    def get_collection_names(self, as_list: bool = False) -> str | List[str]:
        """
        Get all collection names in the database.

        as_list: Return a list of collection names instead of a string. Defaults to False.

        Returns a string of collection names or a list of collection names.
        """
        list_of_collections = []
        for collection in self.chroma_client.list_collections():
            list_of_collections.append(collection.name)

        if as_list:
            return list_of_collections
        
        collection_names = ", ".join(list_of_collections)
        return collection_names