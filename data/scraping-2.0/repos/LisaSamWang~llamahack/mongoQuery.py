from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import cohere
import os

load_dotenv()

uri = os.getenv("MONGO_URI")
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

dbname = client['llama']
convo_collection = dbname['convos']

co = cohere.Client()


def get_result(query):
#Query
    query_embedding = co.embed(texts=[query], model='small').embeddings[0]

    results = convo_collection.aggregate([
        {
            '$vectorSearch': {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "Embeddings",
                "numCandidates": 3,
                "limit": 3
            }
        }
    ])

    res = []
    for document in results:
        res.append(document)

    return res
