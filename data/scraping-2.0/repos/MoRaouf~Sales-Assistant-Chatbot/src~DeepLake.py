import os
import re
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from typing import Optional


class DeepLakeLoader:
    def __init__(self, org_id:str, dataset_name:str, source_data_path:Optional["str"]):
        """
        Initialize DeepLakeLoader object.

        Args:
            org_id (str): Deep Lake Organization ID.
            dataset_name (str): Dataset name to store data.
            source_data_path (str): Path to text file to be processed & stored in Deep Lake.
        """

        self.org_id = org_id
        self.dataset_name = dataset_name
        self.dataset_path = f"hub://{org_id}/{dataset_name}"
        self.source_data_path = source_data_path
        
        # data splitting process
        self.data = self.split_data()

      # laod existing database
        self.db = self.load_db()
            
    # -------------------------------------------------------------------------------------------------------------    

    def load_db(self):
        """
        Load the database if it already exists.

        Returns:
            DeepLake: DeepLake object.
        """
        return DeepLake(dataset_path=self.dataset_path, embedding_function=OpenAIEmbeddings(), read_only=True)
    
    # -------------------------------------------------------------------------------------------------------------

    def create_db(self):
        """
        Create the database if it does not already exist.

        Returns:
            DeepLake: DeepLake object.
        """
        return DeepLake.from_texts(texts=self.data, embedding=OpenAIEmbeddings(), dataset_path=self.dataset_path)
    
    # -------------------------------------------------------------------------------------------------------------

    def query_db(self, query):
        """
        Query the database for passages that are similar to the query.

        Args:
            query (str): Query string.

        Returns:
            content (list): List of passages that are similar to the query.
        """
        results = self.db.similarity_search(query, k=3)
        content = []
        for result in results:
            content.append(result.page_content)
        return content
    
    # -------------------------------------------------------------------------------------------------------------
    
    def add_data(self, texts):
        """
        Add nw data to database

        Args:
            texts (List[str]): List of Texts to be added to database
        """
        self.db.add_texts(texts)

    # -------------------------------------------------------------------------------------------------------------

    def split_data(self):
        """
        Preprocess the data by splitting it into passages.
        If using a different data source, this function will need to be modified.

        Returns:
            split_data (list): List of passages.
        """
        with open(self.source_data_path, 'r') as f:
            content = f.read()
        split_data = re.split(r'(?=\d+\. )', content) # This is super specific to the default data source! If using a different data source, this will need to be modified.
        if split_data[0] == '':
            split_data.pop(0)
        split_data = [entry for entry in split_data if len(entry) >= 30]
        return split_data









