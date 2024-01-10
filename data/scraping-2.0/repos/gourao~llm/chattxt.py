import os
import sys
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator

import environ
env = environ.Env()
environ.Env.read_env()

OPENAI_API_KEY = env('OPENAI_API_KEY')

if OPENAI_API_KEY == "":
	print("Missing OpenAPI key")
	exit()

print("Using OpenAPI with key ["+OPENAI_API_KEY+"]")

path = sys.argv[1]
if path == "":
	print("Missing document path")
	exit()

#loader = TextLoader("data.txt")
#loader = DirectoryLoader("datasets/simple", glob="*.txt")
loader = DirectoryLoader(path, glob="*")

print("Loading done")

# create an index
index = VectorstoreIndexCreator().from_loaders([loader])

def get_prompt():
	print("Type 'exit' to quit")

	while True:
		prompt = input("Enter a prompt: ")

		if prompt.lower() == 'exit':
			print('Exiting...')
			break
		else:
			try:
				# query the index
				print(index.query(prompt))
			except Exception as e:
				print(e)

get_prompt()
