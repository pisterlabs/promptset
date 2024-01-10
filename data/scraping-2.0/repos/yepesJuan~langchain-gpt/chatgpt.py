import os
import sys

import key
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
''' A vectorstore index is a data structure that stores the text of documents as vectors. 
Vectors are a way of representing text that allows for efficient similarity search.'''
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = key.APIKEY

query = sys.argv[1]
print("Query: ", query)

loader = TextLoader('ml.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query))