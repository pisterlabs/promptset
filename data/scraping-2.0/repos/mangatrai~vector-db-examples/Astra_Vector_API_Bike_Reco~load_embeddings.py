import os, requests, json
from dotenv import load_dotenv, find_dotenv
from astrapy.db import AstraDB
from openai import OpenAI

# Load the .env file
if not load_dotenv(find_dotenv(),override=True):
    raise Exception("Couldn't load .env file")

#declare constant
ASTRA_DB_API_ENDPOINT=os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_DB_APPLICATION_TOKEN=os.getenv('ASTRA_DB_TOKEN')
ASTRA_NAMESPACE=os.getenv('ASTRA_NAMESPACE')
ASTRA_COLLECTION=os.getenv('ASTRA_COLLECTION')
model_id = "text-embedding-ada-002"

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

def create_client(astra_db_api_endpoint,astra_db_token, astra_namespace):
    #Establish Connectivity
    astra_db = AstraDB(
    api_endpoint=astra_db_api_endpoint,
    token=astra_db_token,
    namespace=astra_namespace
    )
    return astra_db

def drop_astra_collection(astra_db, astra_drop_collection):
    response = astra_db.delete_collection(astra_drop_collection)
    print("Collection dropped " + str(response))

def create_astra_collection(astra_db, astra_create_collection):
    astra_collection_obj = astra_db.create_collection(astra_create_collection, dimension=1536)
    print("Collection Created ")
    return astra_collection_obj

def load_data_file():
    #load data from sample json file
    url = os.getenv('REVIEWS_FILE_URL')
    response = requests.get(url)
    bikes = response.json()
    print("Bike file loaded")
    return bikes

def create_load_embeddings(bikes, astra_collection_obj):
    for bike in bikes:
        description = bike['description'].replace(',', '\,')
        description = description.replace('"', '\"')
        desc_embedding = client.embeddings.create(input= description, model=model_id).data[0].embedding
        bike['$vector'] = desc_embedding
        response = astra_collection_obj.insert_one(bike)
        print(json.dumps(response,indent=2))
    #response = astra_collection_obj.insert_many(bikes)

def run_loading_data():
    #Create Connection
    astra_db = create_client(ASTRA_DB_API_ENDPOINT,ASTRA_DB_APPLICATION_TOKEN,ASTRA_NAMESPACE)
    
    #database operations
    drop_astra_collection(astra_db, ASTRA_COLLECTION)
    astra_collection_obj = create_astra_collection(astra_db, ASTRA_COLLECTION)

    bikes = load_data_file()
    
    #call embedding function and load data
    create_load_embeddings(bikes, astra_collection_obj)

run_loading_data()




