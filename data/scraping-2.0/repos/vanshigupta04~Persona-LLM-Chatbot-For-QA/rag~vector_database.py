# Description: This file contains the VectorDatabase class, which is used to store and search for vectors in Pinecone.
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
# from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
import pinecone
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

import json
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import CohereEmbedder
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings


class VectorDatabase:
    def __init__(self, embeddings = None, api_key = None, env = None, index_name = None, cohere_api_key = None):
        if api_key is None:
            api_key = os.environ.get("PINECONE_API_KEY")
            print('PINECONE: Loaded API key from environment variables.')
        if env is None:
            env = os.environ.get("PINECONE_ENV")
            print('PINECONE: Loaded environment from environment variables.')
        if index_name is None:
            index_name = os.environ.get("PINECONE_INDEX")
            print('PINECONE: Loaded index name from environment variables.')
        if cohere_api_key is None:
            cohere_api_key = os.environ.get("COHERE_API_KEY")
            print('COHERE: Loaded API key from environment variables.')

        pinecone.init(api_key=api_key, environment=env)
        print('PINECONE: initialized')

        # if index_name not in pinecone.list_indexes():
        #     pinecone.create_index(name=index_name, metric="cosine", dimension=384)
        
        self.index = pinecone.Index(index_name)
        print('PINECONE: Set index to - ', index_name)
        
        if embeddings == None:
            self.embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                model_kwargs={"device": "cuda"}
            )
        else:
            self.embeddings = embeddings
        print('COHERE: loaded embeddings')
        
        self.vector_search = Pinecone(self.index, self.embeddings.embed_query, "text")
        
    def search(self, query, top_k=128):
        return self.vector_search.similarity_search(query, k=top_k)

    def upsert(self, data_path: str):
        """Upserts data into the vector database.

        Args:
            data_path (str): Path to the data file.
        """

        if '.json' in data_path:
            data = pd.read_json(data_path, lines=True, orient='records').to_dict('records')
        elif '.csv' in data_path:
            data = pd.read_csv(data_path).to_dict('records')
        else:
            raise Exception('Data format not supported. Please provide a json or csv file.')

        for item in tqdm(data, desc="Processing data", unit="row", ncols=80):
            conversation_id = item.get("conversation_id")
            speaker = item.get("speaker")
            season = item.get("season")
            episode = item.get("episode")
            scene = item.get("scene")
            text = item.get("text")

            if conversation_id and speaker and text and season and episode and scene:
                metadata = {
                    "speaker": speaker,
                    "season": season,
                    "episode": episode,
                    "scene": scene
                }

                vector = self.embeddings.embed_query(text)
                
                record_metadata = {
                    "text": text, 
                    "source": str(metadata)
                }

                # Perform a single upsert operation with all the data
                self.index.upsert(vectors=[{'id': conversation_id, 'values': vector, 'metadata': record_metadata}])

        print(f"Upserted {len(data)} vectors into the database.")
        
        