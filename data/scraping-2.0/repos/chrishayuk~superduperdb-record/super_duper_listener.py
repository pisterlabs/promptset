from pymongo import MongoClient
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb import Listener, VectorIndex
from superduperdb.ext.openai.model import OpenAIEmbedding
from dotenv import load_dotenv
import os
import json

# load environment variables
load_dotenv()

# load the mongodb connection string
mongo_uri = os.getenv('MONGODB_URI')

# connect to mongodb
client = MongoClient(mongo_uri)

# set the database
db = superduper(MongoClient(mongo_uri).ai_agents)

# collection
collection = Collection('agents')

# open ai connection
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# set the model
model = OpenAIEmbedding(model='text-embedding-ada-002')

db.add(
    VectorIndex(
        identifier='ai-agent-vectors',
        indexing_listener=Listener(
            select=collection.find(),
            key='search_data',
            model=model,
            predict_kwargs={'max_chunk_size':1000},
        )
    )
)
