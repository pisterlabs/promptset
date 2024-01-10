from astrapy.db import AstraDB

import openai, os, uuid, requests
import pandas as pd

from traceloop.sdk import Traceloop
from traceloop.sdk.tracing import tracing as Tracer
from traceloop.sdk.decorators import workflow, task, agent
from dotenv import load_dotenv, find_dotenv

# Load the .env file
if not load_dotenv(find_dotenv(),override=True):
    raise Exception("Couldn't load .env file")

#Add Telemetry
TRACELOOP_API_KEY=os.getenv('TRACELOOP_API_KEY')
Traceloop.init(app_name="Bike Recommendation App", disable_batch=True)
# Generate a UUID
uuid_obj = str(uuid.uuid4())
Tracer.set_correlation_id(uuid_obj)

#declare constant
ASTRA_DB_APPLICATION_TOKEN=os.getenv('ASTRA_DB_APPLICATION_TOKEN')
ASTRA_DB_API_ENDPOINT=os.getenv('ASTRA_DB_API_ENDPOINT')
ASTRA_COLLECTION=os.getenv('ASTRA_COLLECTION')

openai.api_key = os.getenv('OPENAI_API_KEY')
model_id = "text-embedding-ada-002"
vector_len = len(openai.embeddings.create(input="This is to generate an embedding to count the dimension.", model=model_id).data[0].embedding)

@task(name="Establish Astra DB Connection")
def create_connection():
    #Establish Connectivity
    astra_db = AstraDB(token=ASTRA_DB_APPLICATION_TOKEN, api_endpoint=ASTRA_DB_API_ENDPOINT)
    return astra_db

@task(name="Refresh Collection")
def refresh_collection(astra_db):
    # Check whether the collection exists
    collection_list = astra_db.get_collections()
    print("Existing Collections: " + str(collection_list))

    if ASTRA_COLLECTION not in collection_list['status']['collections']:
        # Create the collection
        collection = astra_db.create_collection(ASTRA_COLLECTION, dimension=vector_len)
        print("Collection Created: " + ASTRA_COLLECTION)
    else:
        # Truncate the collection
        collection = astra_db.truncate_collection(ASTRA_COLLECTION)
        print("Collection Truncated: " + ASTRA_COLLECTION)
    return collection


@task(name="Download Raw json data file from source")
def load_data_file():
    #load data from sample json file
    url = "https://raw.githubusercontent.com/ykimoto/HybridWithAstrapy/master/data/bikes.json"
    header= {"content-type": "application/json"}
    response = requests.get(url, headers=header)
    bikes = response.json()
    bikes = pd.DataFrame(bikes)
    return bikes

@task(name="Create and load Embeddings")    
def create_load_embeddings(bikes, collection):

    _id = []
    vector = []

    for id in bikes.index:
        _id.append(id)
        description = bikes['description'][id].replace(',', '\,')
        description = description.replace('"', '\"')

        full_chunk = bikes['description'][id]
        vector.append(openai.embeddings.create(input=full_chunk, model=model_id).data[0].embedding)

    bikes['vector'] = vector
    bikes['_id'] = _id
    cols = bikes.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    bikes = bikes[cols]

    bikes_json = bikes.to_json(orient='records', force_ascii=True)

    # Dec 29, 2023: This is failing at the moment.  Need to investigate.
    #
    # Load the data into the collection
    # res = collection.insert_many(bikes_json)
    # return res
    
    # Resorting to generate a json file to upload and ingest via the Astra UI
    with open('bikes_withVector.json', 'w') as f:
        f.write(bikes_json)
    
    return("OK")
    
    


@workflow(name="Load Bike Recommendation Data")
def run_loading_data():
    #Create Connection
    astra_db = create_connection()
    
    #Create or Truncate the Bike Catalog Collection
    collection = refresh_collection(astra_db)

    bikes = load_data_file()
    
    #call embedding function and load data
    res = create_load_embeddings(bikes, collection)

run_loading_data()