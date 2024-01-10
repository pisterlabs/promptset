import sys

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

from config import load_config

config = load_config()

query = sys.argv[1]
print(query)

# loader = TextLoader('Data.txt')
loader = DirectoryLoader('.', glob='*.txt')

index = VectorstoreIndexCreator().from_loaders([loader])

# print(index.query(query))
print(index.query(query, llm=ChatOpenAI()))
