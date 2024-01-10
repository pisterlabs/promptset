import os
import sys
import constants

from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


os.environ["OPENAI_API_KEY"] = constants.APIKEY


query = sys.argv[1]
print("Your Question: " + query)

#single file 
loader = TextLoader('data/data.txt')

#loading all files in directory
# loader = DirectoryLoader(".", glob="*.txt")

index = VectorstoreIndexCreator().from_loaders([loader])

#without passing in external model, only using vector store
print(index.query(query))
#passing in chatopenai for external model
# print(index.query(query, llm=ChatOpenAI()))/