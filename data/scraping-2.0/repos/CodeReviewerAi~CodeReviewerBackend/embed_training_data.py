import openai
import requests
import json
import os
import random
import time
import dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import CollectionDescription, Distance, VectorParams, Record

def embed_repos_functions(json_data):
    client = QdrantClient(host='localhost', port=6333)
    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = "text-embedding-ada-002"

    if not client.get_collections().collections.__contains__(CollectionDescription(name='functions')):
        client.create_collection(
            collection_name="functions",
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE
            )
        )

    for repo_name, functions in json_data.items():
        for function_key, function_data in functions.items():
            merged_function = function_data['merged_function']

            print(f"Processing function '{function_key}' from repo '{repo_name}'")

            attempt = 0
            max_attempts = 5
            while attempt < max_attempts:
                try:
                    response = requests.post(
                        'https://api.openai.com/v1/embeddings',
                        headers={
                            'Content-Type': 'application/json',
                            'Authorization': f'Bearer {openai.api_key}'
                        },
                        data=json.dumps({
                            "input": merged_function,
                            "model": model
                        }),
                        timeout=30  # Timeout for the request
                    )

                    if response.status_code != 200:
                        raise Exception(f"Failed to embed: Status {response.status_code}, Response: {response.text}")

                    embedding = response.json()['data'][0]['embedding']
                    id = random.randint(0, 1000000)
                    payload = {
                        "function_key": function_key,
                        "score": function_data.get('score', 0),
                    }

                    print(f"Uploading function '{function_key}' to Qdrant")

                    client.upload_records(
                        collection_name="functions",
                        records=[
                            Record(
                                id=id,
                                vector=embedding,
                                payload=payload
                            )
                        ]
                    )
                    print(f"Added function '{function_key}' from repo '{repo_name}' to Qdrant with ID {id}")
                    break

                except Exception as e:
                    print(f"Exception occurred for '{function_key}' from '{repo_name}': {str(e)}. Retrying in 30 seconds...")
                    time.sleep(30)
                    attempt += 1

            if attempt == max_attempts:
                print(f"Failed to process '{function_key}' from '{repo_name}' after {max_attempts} attempts.")

