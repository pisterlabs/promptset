import random
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
import openai
import os
import csv

openai.api_key = 'your_openai_api_key'
qdrant_api_key = 'your_qdrant_api_key'
cohere_api_key = 'your_cohere_api_key'



os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
os.environ['COHERE_API_KEY'] = cohere_api_key
os.environ['QDRANT_API_KEY'] = qdrant_api_key


Path.ls = lambda x: list(x.iterdir())
random.seed(42)  # This is the answer

qdrant_client = QdrantClient(
    url="quadrant_URL",
    api_key=qdrant_api_key,
)








def check_environment_keys():
    """
    Utility Function that you have the NECESSARY Keys
    """
    if os.environ.get('OPENAI_API_KEY') is None:
        raise ValueError(
            "OPENAI_API_KEY cannot be None. Set the key using os.environ['OPENAI_API_KEY']='sk-xxx'"
        )
    if os.environ.get('COHERE_API_KEY') is None:
        raise ValueError(
            "COHERE_API_KEY cannot be None. Set the key using os.environ['COHERE_API_KEY']='xxx'"
        )
    if os.environ.get("QDRANT_API_KEY") is None:
        print("[Optional] If you want to use the Qdrant Cloud, please get the Qdrant Cloud API Keys and URL")


check_environment_keys()




def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Function to create embeddings from chunks of text
def create_embeddings(text_chunks):
    points = []
    i = 0
    for chunk in text_chunks:
        i += 1
#         print("Embeddings chunk:", chunk)
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embeddings = response['data'][0]['embedding']
        points.append(PointStruct(id=i, vector=embeddings, payload={"text": chunk}))
    return points

# Replace 'your_input_file.csv' with your actual CSV file name
input_file = 'bigBasketProducts.csv'


text_data = []
with open(input_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    num_readings = 2 
    for index, row in enumerate(reader):
        if index >= num_readings:
            break
        text_data.extend(row)

chunk_size = 10  
text_chunks = list(chunks(text_data, chunk_size))

resulting_points = create_embeddings(text_chunks)

qdrant_client.upsert(collection_name='product', wait=True, points=resulting_points)


def create_answer_with_context(query):
    response = openai.Embedding.create(
        input="write details of Garlic Oil - Vegetarian Capsule 500 mg",
        model="text-embedding-ada-002"
    )
    embeddings = response['data'][0]['embedding']

    search_result = qdrant_client.search(
        collection_name="product",
        query_vector=embeddings,
        limit=5
    )
    
    prompt = "Context:\n"
    
    for result in search_result:
      
        text = ' '.join(result.payload['text']) if isinstance(result.payload['text'], list) else result.payload['text']
        prompt += text + "\n---\n"
    prompt += "Question:" + query + "\n---\n" + "Answer:"



    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
        )

    return completion.choices[0].message.content

query = input("write your query : ")
answer = create_answer_with_context(query)
print(f"Answer:\n {answer}")