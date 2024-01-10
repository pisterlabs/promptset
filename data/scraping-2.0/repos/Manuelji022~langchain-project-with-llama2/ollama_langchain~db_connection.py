import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch

class DBConnection():
    """
    Class for connecting to MongoDB Atlas and upload the embeddings.
    """
    def __init__(self):
        load_dotenv()
        self.connection_string = os.getenv("MONGODB_CONNECTION")
        self.db_name = os.getenv("_DB_NAME")
        self.collection_name = os.getenv("COLLECTION_NAME")
        self.index_name = os.getenv("INDEX_NAME")
    
    def connect_to_db(self):
        """Connect to MongoDB Atlas."""
        try:
            self.client = MongoClient(self.connection_string)
            self.collection = self.client[self.db_name][self.collection_name]
        except Exception as e:
            logging.error(f"An error ocurred while connecting to the database {e}")

    def close_connection(self):
        """Close the connection to MongoDB Atlas."""
        if self.client is not None:
            self.client.close()

    def upload_embeddings(self, chunks, embeddings):
        """Upload the embeddings to MongoDB Atlas."""
        self.docsearch = MongoDBAtlasVectorSearch.from_documents(
            chunks, embeddings, collection=self.collection, index_name=self.index_name
        )
    