from langchain_community.document_loaders import JSONLoader
import os
import logging
from langchain_community.document_loaders import PyPDFLoader

class DataLoader:
    def __init__(self, data_path, content_key=None, metadata_cols=None, loader_type='json'):
        self.data_path = data_path
        self.loader_type = loader_type
        self.content_key = content_key
        self.metadata_cols = metadata_cols

    def metadata_func(record: dict, metadata: dict) -> dict:
        for col in self.metadata_cols:
            metadata[col] = record.get(col)

        return metadata
        
    def load(self):
        if self.loader_type == 'json':
            return JSONLoader(file_path=self.data_path,
                              content_key=self.content_key,
                              metadata_func=metadata_func).load()
        
        elif self.loader_type == 'pdf':
            return PyPDFLoader(file_path=self.data_path).load_and_split()
        
        #Use same loader class for different data types later
        else:
            raise Exception('Loader type not supported')