import os
import json
import requests
import pinecone
import openai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# Load OpenAI API key and Pinecone API key from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_UPSERT_URL = os.getenv("PINECONE_UPSERT_URL")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE")

# Initialize OpenAI API and PineCone
openai.api_key = OPENAI_API_KEY
pinecone.init(api_key=PINECONE_API_KEY)

# Replace this with the ID of your Pinecone namespace
pinecone_namespace = PINECONE_NAMESPACE


def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-ada-002")
    return response["data"][0]["embedding"]


def upsert_embedding_to_pinecone(id, embedding):
    upsert_url = PINECONE_UPSERT_URL
    headers = {
        "Content-Type": "application/json",
        "Api-Key": PINECONE_API_KEY,
    }
    data = {
        "vectors": [
            {
                "id": id,
                "metadata": {},
                "values": embedding,
            }
        ],
        "namespace": pinecone_namespace,
    }
    response = requests.post(upsert_url, headers=headers, data=json.dumps(data))
    return response


def process_file(file_name):
    input_file_path = os.path.join(input_folder, file_name)

    with open(input_file_path, "r", encoding="utf-8") as file:
        text_content = file.read()

    embedding = get_embedding(text_content)
    response = upsert_embedding_to_pinecone(file_name, embedding)

    print(f"Processed {input_file_path} and saved embedding to Pinecone with status code {response.status_code}")


input_folder = "chunks"

with ThreadPoolExecutor(max_workers=10) as executor:
    txt_files = [file_name for file_name in os.listdir(input_folder) if file_name.endswith(".txt")]
    executor.map(process_file, txt_files)
