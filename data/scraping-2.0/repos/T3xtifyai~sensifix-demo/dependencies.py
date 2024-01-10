from pymongo import MongoClient

from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

#loading the environment variables
load_dotenv()

#loading mongodb connection uri
mongodb_uri = os.getenv("MONGO_DB_URI")

#connecting to MongoDB client
client = MongoClient(mongodb_uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You are successfully connected to MongoDB!")
except Exception as e:
    print(e)

# loading our database
db_name = os.getenv("DB")
db = client[db_name]

# loading the names of collections(tables)
collection_cl1_name = os.getenv("COLLECTION_CAT_L1")
collection_cl2_name = os.getenv("COLLECTION_CAT_L2")
collection_val = os.getenv("COLLECTION_VALID")
collection_ticket = os.getenv("COLLECTION_TICKET")
collection_resp = os.getenv("COLLECTION_RESP")

#loading translation api ley
translate_key = os.getenv("TRANSLATE_KEY")

# loading the llms
llm = OpenAI(model_name="gpt-3.5-turbo-0301", temperature=0.5, top_p=0.8)