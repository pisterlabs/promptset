import os
import re
import sys
import environ
import pdb
import json
import jsonschema

import sqlite3
import psycopg2

from jsonschema import Draft7Validator

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# prompt loop
def get_prompt(chain):
	print("Type 'exit' to quit")

	while True:
		prompt = input("Enter a prompt: ")

		if prompt.lower() == 'exit':
			print('Exiting...')
			break
		else:
			try:
				pass
			except Exception as e:
				print(e)

def create_schema(cursor, schema, table_name=''):
	print("Processing JSON: ", schema)

	column_types = {}

	if isinstance(schema, dict):
		# If JSON data is a dictionary, recursively process its values
		for key, value in schema.items():
			key = re.sub(r'[^a-zA-Z0-9_]', '_', key)
			if isinstance(value, (dict, list)):
				# For nested dictionaries or lists, create a separate table
				create_schema(cursor, value, table_name + "_" + key)
			else:
				# Insert primitive values as a column
				if value == "int":
					column_types[key] = 'INTEGER'
				elif value == "float":
					column_types[key] = 'REAL'
				elif value == "bool":
					column_types[key] = 'BOOL'
				else:
					column_types[key] = 'TEXT'
	elif isinstance(schema, list):
		# If JSON data is a list, iterate over its elements
		for item in schema:
			if isinstance(item, (dict, list)):
				# For nested dictionaries or lists, create a separate table
				create_schema(cursor, item, table_name)
			else:
				# Insert primitive values as a column
				if value == "int":
					column_types[key] = 'INTEGER'
				elif value == "float":
					column_types[key] = 'REAL'
				elif value == "bool":
					column_types[key] = 'BOOL'
				else:
					column_types[key] = 'TEXT'

	# Create SQL table if there are any primitive columns to add
	if len(column_types) > 0:
		create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ("
		create_table_sql += ", ".join([f"{key} {column_types[key]}" for key in column_types])
		create_table_sql += ");"

		print("Executing: ", create_table_sql)
		cursor.execute(create_table_sql)

def import_data(cursor, json_data_path):
	pass

env = environ.Env()
environ.Env.read_env()

# Load API key and document location
OPENAI_API_KEY = env('OPENAI_API_KEY')

if OPENAI_API_KEY == "":
	print("Missing OpenAPI key")
	exit()

print("Using OpenAPI with key ["+OPENAI_API_KEY+"]")

json_data_path = sys.argv[1]
if json_data_path == "":
	print("Missing document path")
	exit()

schema_file_path = sys.argv[2]
if schema_file_path == "":
	print("Missing schemat path")
	exit()

# Read JSON Schema
with open(schema_file_path, 'r') as schema_file:
	schema = json.load(schema_file)

# Print the JSON schema
print(json.dumps(schema, indent=2))

prompt = input("Hit Enter to Add Data to SQL")

# Create SQL schema and import data

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

create_schema(cursor, schema, "main")
conn.commit()
prompt = input("Schema created")

import_data(cursor, json_data_path)
prompt = input("Data imported into SQL")

# setup llm
llm = ChatOpenAI(model_name="gpt-3.5-turbo",
	temperature=0.7,
	# max_tokens=1024,
	openai_api_key=OPENAI_API_KEY)

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

get_prompt()
