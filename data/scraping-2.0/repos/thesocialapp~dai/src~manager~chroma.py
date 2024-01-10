from chromadb.utils import embedding_functions
from src.manager.document import Document
from decouple import config
from typing import List
import chromadb
from langchain.embeddings import OpenAIEmbeddings

class ChromaDBManager:
    _instance = None
    
    def __new__(cls):
        mode = config('ENVIRONMENT')
        settings = chromadb.Settings(
            is_persistent=True,
            allow_reset=True,
        )
        if cls._instance is None:
            cls._instance = super(ChromaDBManager, cls).__new__(cls)
            cls._instance.openEmb = OpenAIEmbeddings(
                api_key=config('OPENAI_API_KEY'),
            )
            cls._instance.sentenceEmb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            if mode == 'development':
                cls._instance._chromaDb = chromadb.PersistentClient(
                    settings=settings,
                )
            else:
                host = config('CHROMADB_HOST', 'chromadb')
                port = config('CHROMADB_PORT', 8000, cast=int)
                cls._instance._chromaDb = chromadb.HttpClient(
                    host=host,
                    port=port,
                    settings=settings,
                )

            
        return cls._instance
    
    @property
    def chromaDb(self):
        return self._chromaDb
    
    
    def add_document(self, collection_name, document: Document) -> tuple[bool, str]:
        """
        Add a document to the specified collection in ChromaDB.

        Parameters:
            - collection_name (str): The name of the collection.
            - document (dict): The document to be added.

        Returns:
            - bool: True if the document is successfully added, False otherwise.
        """
        # Get collection or create it
        # Add the document to the collection
        try:
            collection = self._chromaDb.get_or_create_collection(collection_name, embedding_function=self._instance.sentenceEmb)
            return self.embedd_and_add_document(collection, document)
        except Exception as e:
            print(f'Error adding document to collection {collection_name}: {e}')
            return False, str(e)
    
    
    
    
    def embedd_and_add_document(self, collection, document: Document) -> tuple[bool, str]:
        """
        Add a document to the specified collection in ChromaDB.

        Parameters:
            - collection (Collection): The collection we will add the documents to
            - document (Dict): The document to be added.

        Returns:
            - bool: True if the document is successfully added, False otherwise.
        """
        
        # Add the document to the collection
        try:
            print(f"Document contents {len(document.contents)}")
            print(f"We have embeds")
            # Add the document to the collection
            result = collection.add(
                ids=document.ids,
                metadatas=document.metadata,
                documents=document.contents,
            )
            print(f'We have the results in {result}')
            return True, None
        except Exception as e:
            print(f'We have a problem {e}')
            return False, str(e)

        
    def get_documents(self, collection_name):
        """
        Get a document from the specified collection in ChromaDB.

        Parameters:
            - collection_name (str): The name of the collection.

        Returns:
            - tuple: The document if it exists, None otherwise.
        """

        try:
            # Get collection or create it
            print(f"Getting collection {collection_name}")
            collection = self._chromaDb.get_collection(collection_name)
            docs = collection.get()
            print('We have the docs')
            return docs, None
        except Exception as e:
            print(f'Error getting document collection {collection_name}: {e}')
            return None, str(e)
        
    def update_document(self, collection_name, document: Document) -> tuple[bool, str]:
        """
        Update a document in the specified collection in ChromaDB.

        Parameters:
            - collection_name (str): The name of the collection.
            - document (Document): The document will be recreated to allow the current 
            stored document to be updated.

        Returns:
            - bool: True if the document is successfully updated, False otherwise.
        """

        # Update the document in the collection
        try:
            # We need to search for documents that match the document ids
            # and update later on, this is to prevent the extra step of creating new embeddings
            # for the document
            collection = self._chromaDb.get_collection(collection_name)
            collection.update(
                ids=document.ids,
                documents=document.content,
            )
            return True, None
        except Exception as e:
            print(f'Error updating document in collection {collection_name}: {e}')
            return False, str(e)
        
    def clear_collection(self, collection_name):
        """
        Clear a collection in ChromaDB.

        Parameters:
            - collection_name (str): The name of the collection.

        Returns:
            - bool: True if the collection is successfully cleared, False otherwise.
        """

        try:
            # Delete collection
            self._chromaDb.delete_collection(collection_name)
            return True, None
        except Exception as e:
            print(f'Error clearing collection {collection_name}: {e}')
            return False, str(e)
        
    def reset(self):
        try:
            self._chromaDb.reset()
            return True
        except Exception as e:
            print(e)
            return False