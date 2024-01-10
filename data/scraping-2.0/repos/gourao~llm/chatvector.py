import os
import json
import sys
import environ

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import OpenAIEmbeddings

from langchain.vectorstores import Chroma
from langchain.vectorstores import Milvus

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

env = environ.Env()
environ.Env.read_env()

# Load API key and document location
OPENAI_API_KEY = env('OPENAI_API_KEY')

if OPENAI_API_KEY == "":
	print("Missing OpenAPI key")
	exit()

print("Using OpenAPI with key ["+OPENAI_API_KEY+"]")

path = sys.argv[1]
if path == "":
	print("Missing document path")
	exit()

# Document loading
loader = DirectoryLoader(path, glob="*")
data = loader.load()
print("Loading done")

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(
	chunk_size = 500, 
	chunk_overlap = 0
)

all_splits = text_splitter.split_documents(data)
print("Splitting done")

# Create retriever

vectorstore = Chroma.from_documents(
	documents=all_splits, 
	embedding=OpenAIEmbeddings()
)
'''
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
vectorstore = Milvus.from_documents(
	all_splits,
	embedding=OpenAIEmbeddings(),
    connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT}
)
'''

print("Vector store created")

# prompt loop
def get_prompt():
	print("Type 'exit' to quit")

	while True:
		prompt = input("Enter a prompt: ")

		if prompt.lower() == 'exit':
			print('Exiting...')
			break
		else:
			try:
				docs = vectorstore.similarity_search(prompt)
				print(docs)
				for i in len(docs):
					print(json.dumps(docs[i], indent=2))
			except Exception as e:
				print(e)

get_prompt()
