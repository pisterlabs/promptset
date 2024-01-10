from langchain.tools import BaseTool
from typing import Any
import openai
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import CrossEncoder
import numpy as np
import json
import subprocess

os.environ['TOKENIZERS_PARALLELISM'] = 'true'

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
reservoir_api_key = os.getenv("RESERVOIR_API_KEY") # This key is not currently in use. 

class ReservoirTool(BaseTool):
    name = "Reservoir NFT Tool"
    description = '''Use this tool when you need to find specific information about an NFT collection after you have found the right address for it.
    Use the URL of the collection's metadata file based on the chain that is being used, defaulting to api.reservoir.tools for Ethereum.    
    The terms "address" and "collection names" are interchangeable. If an endpoint requires a collection name, use the address as the collection.
    '''
    model = "gpt-4"

    def get_url(self):
        client = chromadb.PersistentClient(path='../chroma.db')
        collection = client.get_collection(name='reservoir')
        all_collections = [collection.metadata]
        return all_collections  # Method should return value

    def get_endpoints(self, task: str):
        # Initialize cross-encoder
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')

        # Create database client
        client = chromadb.PersistentClient(path='../chroma.db')

        collection_name = "reservoir"

        # Use the task as the query
        query_text = task

        restructured = []
        pairs = []  # Store all pairs for cross-encoder

        # Get the collection
        collection = client.get_collection(name=collection_name)

        # Query the collection
        data = collection.query(
            query_texts=[query_text],
            n_results=10
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

        return result[:1]

    def use_endpoint(self, url, endpoints, address):

        query_message = {
            "role": "user",
            "content": f"Given the URL '{url}', endpoint {endpoints}, and the address {address}, write a curl request to the API. Do not add placeholders, use the information given."
        }

        response = openai.ChatCompletion.create(
                model=self.model,
                messages=[query_message],
                temperature=0.1,  # Lower temperature to make output more deterministic
                max_tokens=150  # Limit output to 100 tokens
            )

        request_info = response['choices'][0]['message']['content']
        return request_info

    
    def _run(self, task: str, address: str = None):
        url = self.get_url()  # Get URL
        endpoints = self.get_endpoints(task)  # Get endpoints based on task
        request = self.use_endpoint(url, endpoints, address)  # Use endpoint to get request

        request_message = {
            "role": "user",
            "content": f"Given the output of the last model {request}, delete everything but the curl request."
        }

        curl = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[request_message],
                temperature=0,
                max_tokens=200
            )

        curl = curl['choices'][0]['message']['content']
        print(curl)

        process = subprocess.Popen(curl, stdout=subprocess.PIPE, shell=True)
        output, error = process.communicate()

        if error:
            response = f"Error: {error}"
        else:
            response = f"Output: {output.decode('utf-8')}"

        # Truncate response to approximately 18,000 characters if needed
        if len(response) > 15000:
            response = response[:14997] + "..."

        return response


    def _arun(self, task: Any):
        raise NotImplementedError("This tool does not support async")
