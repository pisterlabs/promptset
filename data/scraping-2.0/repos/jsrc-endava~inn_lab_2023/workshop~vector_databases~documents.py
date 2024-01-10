import os
from dotenv import load_dotenv
from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv('.env')

# Langchain allows you to use any vector database, it has multiple built-in
langchain_milvus = Milvus(
  embedding_function=OpenAIEmbeddings(),
  collection_name="innovation_lab",
  connection_args={
    "uri": os.environ['ZILLIZ_ENDPOINT'],
    "token": os.environ['ZILLIZ_TOKEN'],
  },
)

print(langchain_milvus.add_texts(
  texts=[
    "Hello world!",
    "Synaptron team!",
  ],
  metadatas=[
    {"some_id": 1},
    {"some_id": 2},
  ]
))
