from langchain.tools import BaseTool
from typing import Any
import openai
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import CrossEncoder
import numpy as np
import json

# This is an example of a tool that uses the ChromaDB database to find the right endpoint for a given task. It can be used as a template

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class GetEndpointsTool(BaseTool):
    name = "Endpoints tool"
    description = "Use this tool when you need to find the right endpoint for a given task. Only input the task, no other information is needed."
    model = "gpt-4"

    def _run(self, task: str):
        # Initialize cross-encoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')

        # Create database client
        client = chromadb.PersistentClient(path='../chroma.db')

        # Get a list of all collection names in the database.
        all_collections = [collection.name for collection in client.list_collections()]

        # Use the task as the query
        query_text = task

        restructured = []
        pairs = []  # Store all pairs for cross-encoder

        # Loop over each collection
        for collection_name in all_collections:
            # Get the collection
            collection = client.get_collection(name=collection_name)

            # Query the collection
            data = collection.query(
                query_texts=[query_text],
                n_results=5
            )

            # Restructure the data and store sentence pairs
            for i in range(len(data['ids'][0])):
                document = json.loads(data['documents'][0][i])  # Convert 'document' from string to dictionary.
                generated_description = document.get('generated_description')
                
                pairs.append([query_text, generated_description])  # Add each pair to the list
                
                restructured.append({
                    'collection': collection_name,  # Include the name of the collection in the restructured data.
                    'id': data['ids'][0][i],
                    'document': document,
                })

        # Compute the similarity scores for the pairs
        similarity_scores = model.predict(pairs)

        # Add the similarity score to the corresponding document in the restructured list
        for i, score in enumerate(similarity_scores):
            restructured[i]['similarity_score'] = score

        # Sort the restructured list based on similarity scores in descending order
        restructured.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Aggregate the sorted restructured list and return
        result = []
        for item in restructured:
            result.append(f"ID: {item['id']}, Collection: {item['collection']}, Document: {item['document']}, Similarity Score: {item['similarity_score']}")
        
        return result

    def _arun(self, task: Any):
        raise NotImplementedError("This tool does not support async")
