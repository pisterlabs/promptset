
from dotenv import load_dotenv
import pinecone
import openai
import os
import json
from time import time 
import tqdm
# Typing
from typing import List, Dict, Any, Optional, Union, Tuple

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone.index import Index
from pinecone.index import UpsertResponse

# Data processing stuff
import pandas as pd

opeai_embedding_model = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()

class DataEmbedding:
    """Main Wrapper class to handle data embedding,from different data sources/formats"""

    def __init__(self,openai_embedding: OpenAIEmbeddings, text_splitter: RecursiveCharacterTextSplitter):
        self.openai_embedding = openai_embedding
        self.text_splitter = text_splitter

    @staticmethod
    def _formatMetadata(metadata:str)->dict:
        """Takes in a string from .to_json method and returns a dictionary with nulls processed"""
        json_metadata = json.loads(metadata)
        # Change all nulls to Null:str
        for key,value in json_metadata.items():
            if value is None:
                json_metadata[key] = "Null"

        return json_metadata
    
    def prepareExcel(self,df:pd.DataFrame) -> list[dict]:
        """Takes in a dataframe and returns a list of dictionaries in the form of
        {'id': str, 'values': List[float], 'sparse_values': {'indices': List[int], 'values': List[float]},
            'metadata': dict}
        """
        # each row is a dict
        data = []

        for index, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
            # Extract all the values from the row and store them in a single string, separated by a comma E.g Week: 52,Work Center: Packr 8H,
            row_text = row.to_json(date_format='iso')
            values = self.openai_embedding.embed_query(row_text)
            data.append({
                'id': str(index),
                'values': values,
                'metadata': self._formatMetadata(row_text)
            })
        return data
    
    def prepareJson(self,QNA_list:list[dict])->list[dict]:
        """Takes in list of dict with keys 'question', 'answers' 
        {'id': str, 'values': List[float], 'sparse_values': {'indices': List[int], 'values': List[float]},
            'metadata': dict}
        """
        # each row is a dict
        data = []

        for index, row in tqdm.tqdm(enumerate(QNA_list), total=len(QNA_list)):
            # Extract all the values from the row and store them in a single string, separated by a comma E.g Week: 52,Work Center: Packr 8H,
            list_of_questions = row['questions']
            # concat to a single string
            row_text = ','.join(list_of_questions)
            
            values = self.openai_embedding.embed_query(row_text)
            data.append({
                'id': str(index),
                'values': values,
                'metadata': row
            })
        return data
        
        

class Indexer:
    """Main class to upsert both text/sentence/images with pinecone"""

    dataEmbeddings = DataEmbedding(opeai_embedding_model,text_splitter)
    
    @staticmethod
    def _initPinecone(PINECONE_API_KEY:str,PINECONE_ENVIRONMENT:str,INDEX_NAME:str) -> pinecone.index:
        """Init Pinecone stuff"""

        pinecone.init(api_key=PINECONE_API_KEY,environment=PINECONE_ENVIRONMENT)
        # connect to index
        index:pinecone.index = pinecone.Index(INDEX_NAME)
        print(f'Successfully connected to Pinecone Index:\n{index.describe_index_stats()}')
        return index

    def __init__(self,PINECONE_API_KEY:str,PINECONE_ENVIRONMENT:str,INDEX_NAME:str):
        """ Initialize the Pinecone Index from .env file pinecone variables"""
        # connect to index
        self.index:Index = self._initPinecone(PINECONE_API_KEY,PINECONE_ENVIRONMENT,INDEX_NAME)
    
    def upsertExcel(self, df, batch_size=100):
        # Split the dataframe into batches
        """Upsert by row from a pandas dataframe"""
        data_batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]

        upsert_responses = []
        for batch in data_batches:
            data = self.dataEmbeddings.prepareExcel(batch)
            upsert_response = self.index.upsert(data)
            upsert_responses.append(upsert_response)

        return upsert_responses
    
    def upsertQNA(self,QNA_list:list[dict],batch_size=10):
        """Upsert by list of dict Question, Answer pairing"""
        data = self.dataEmbeddings.prepareJson(QNA_list)

        data_batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        print('data_batches: ', data_batches)

        upsert_responses = []
        for batch in data_batches:
            upsert_response = self.index.upsert(batch)
            upsert_responses.append(upsert_response)
            print(f'Upserted {len(data)} rows, upsert_response: {upsert_response}',flush=True)

        return upsert_responses

    