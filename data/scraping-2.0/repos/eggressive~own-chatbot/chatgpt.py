import os
import sys
import openai
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

openai_api_key = os.environ.get('OPENAI_API_KEY')

query = sys.argv[1]
#print(query)

loader = TextLoader('data/data.txt')
index = VectorstoreIndexCreator().from_loaders([loader])

print(index.query(query, llm=ChatOpenAI(model="gpt-3.5-turbo")))