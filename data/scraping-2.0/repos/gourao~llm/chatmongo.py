import os
import sys
import environ
import pdb

import pymongo
import json

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

env = environ.Env()
environ.Env.read_env()

API_KEY = env('OPENAI_API_KEY')

if API_KEY == "":
	print("Missing OpenAPI key")
	exit()

mongo_uri = sys.argv[1]
if mongo_uri == "":
	print("Missing MongoDB URI")
	exit()

db_name = sys.argv[2]
if db_name == "":
	print("Missing DB")
	exit()

collection_name = sys.argv[3]
if collection_name == "":
	print("Missing Collection")
	exit()

schema_name = sys.argv[4]
if schema_name == "":
	print("Missing Schema")
	exit()

with open(schema_name, 'r') as json_file:
	schema = json.dumps(json.load(json_file), separators=(',', ':'))
	print(schema)

print("Using :", mongo_uri, ":", "["+db_name+"]")

# setup llm
llm = ChatOpenAI(model_name="gpt-3.5-turbo",
	temperature=0.7,
	# max_tokens=1024,
	openai_api_key=API_KEY)

# DEBUG
dbg_prompt = PromptTemplate(
    input_variables=["schema"],
    template="""Given the schema below construct a json nosql query to find all volumes with three replicas

				{schema}
			"""
)
dbg_chain = LLMChain(llm=llm, prompt=dbg_prompt)

# pdb.set_trace()
print(dbg_chain.run({
	'schema': schema
}))

exit()

# Create prompt chain
prompt = PromptTemplate(
    input_variables=["collection", "schema", "question"],
    template="""Using the Schema Below, create a syntactically correct Mongo NoSQL query that matches the schema to run.  First is the collection, followed by the schema and then the question.
				Collection: {collection}
				Schema: {schema}
				Question: {question}
			"""
)

chain = LLMChain(llm=llm, prompt=prompt)

def chatmongo(collection):
	print("Entering chat session with ", collection.name)
	print("Type 'exit' to quit")

	while True:
		prompt = input("Enter a prompt: ")

		if prompt.lower() == 'exit':
			print('Exiting...')
			break
		else:
			print(chain.run({
				'collection': collection.name,
				'schema': schema, 
				'question': prompt
				}))

try:
	# Connect to MongoDB
	client = pymongo.MongoClient(mongo_uri)

	# List all available databases
	database_names = client.list_database_names()

	# Print the list of database names
	print("Databases in the cluster:")
	for name in database_names:
		print(name)

	db = client[db_name]

	# Get a list of collection names in the database
	collection_names = db.list_collection_names()

	if collection_names:
		print("Collections in the database: ", db_name)
		for collection in collection_names:
			print(collection)

		collection = db.collections[collection_name]
		chatmongo(collection)
	else:
		print("No collections found in the database.")

except Exception as e:
	print(f"An error occurred: {e}")

finally:
	# Close the MongoDB connection when done
	client.close()

