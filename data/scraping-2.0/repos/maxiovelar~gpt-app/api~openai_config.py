import os
import shutil
import json
import requests
import tempfile
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from api.pinecone_db import get_db_embeddings

# DATABASE_PATH = './db/'
tmp_dir = tempfile.gettempdir()
DATABASE_PATH = tmp_dir + '/db/'

################################################################################
# API Swell
################################################################################
url = os.getenv('SWELL_API_URL')

def handle_db():
    # Define your custom headers here
    headers = {
        'Authorization': os.getenv('SWELL_AUTORIZATION_KEY'),  # If you have an API token or authentication
        'Content-Type': 'application/json',  # If needed for the request body
    }

    # Make the HTTP request with custom headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Load the JSON data from the response
        json_data = response.json()

        # Initialize an empty list to store extracted data for each item
        extracted_data_list = []

        # Loop through each item in the array and extract the desired information
        for item in json_data["results"]:
            if item.get("active"):
                extracted_data = {
                    "id": item.get("id", None),
                    "name": item.get("name", None),
                    "active": item.get("active", None),
                    "description": item.get("description", None),
                    "sale": item.get("sale", None),
                    "sale_price": item.get("sale_price", None),
                    "price": item.get("price", None),
                    "currency": item.get("currency", None),
                    "slug": item.get("slug", None),
                    "stock": item.get("stock_level", None),
                    "image_url": item.get("images")[0].get("file").get("url")
                }
                extracted_data_list.append(extracted_data)    
        
        open('data.json', 'w').write(json.dumps( extracted_data_list, indent=4))
        open('dataSwell.json', 'w').write(json.dumps( json_data, indent=4))
        # Process the JSON data as needed
        print(extracted_data_list)
    else:
        print(f"Request failed with status code: {response.status_code}")

################################################################################
# Read documents from JSON file and add them to pinecone instance
################################################################################
def revalidate():
    handle_db()
    
    if os.path.exists(DATABASE_PATH):
        shutil.rmtree(DATABASE_PATH)
        
    loader = JSONLoader(
        file_path='./data.json',
        jq_schema='.[]',
        text_content=False
    )
    
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators= ["\n\n", "\n", "(?<=\. )", ";", ",", " ", ""]) # se puede pasar regex a los separators
    docs = text_splitter.split_documents(documents)

    get_db_embeddings(docs)