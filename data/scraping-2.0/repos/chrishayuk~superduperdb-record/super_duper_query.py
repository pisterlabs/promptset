from pymongo import MongoClient
from superduperdb import superduper
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document
from superduperdb.ext.openai import OpenAIChatCompletion
from dotenv import load_dotenv
import os

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

prompt_template = (
    "{context}\n\n"
    "Here's the question:\n"
)

chat = OpenAIChatCompletion(model="gpt-3.5-turbo", prompt=prompt_template)
db.add(chat)

question = "what are good queries to interact with timebot uk?"
num_results = 5

output, context = db.predict(
    model_name="gpt-3.5-turbo",
    input=question,
    context_select=(
        collection.like(
            Document({"search_data": question}), vector_index="ai-agent-vectors", n=num_results
        ).find()
    ),
    context_key="search_data",
)

print(question)
print(output.content)
