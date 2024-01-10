# This is a simple script to embed a set of text files located in a data_directory, and store them in a Chroma collection.

import os
import argparse

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


def get_or_create_collection(collection_name: str, persist_directory: str):
  """
  Instantiates the Chroma client, and creates a collection, using OpenAI embeddings.
  """

  # Instantiate a persistent chroma client in the persist_directory.
  # Learn more at docs.trychroma.com
  client = chromadb.Client(settings=Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))

  # We use the OpenAI embedding function.
  embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ['OPENAI_API_KEY'])

  # If the collection already exists, we just return it. This allows us to add more
  # data to an existing collection.
  collection = client.get_or_create_collection(
    name=collection_name, embedding_function=embedding_function)

  return collection


def main(data_dir: str, collection_name: str, persist_directory: str):
  # Read all files in the data directory
  documents = []
  metadatas = []
  files = os.listdir(data_dir)
  for filename in files:
    with open(f'{data_dir}/{filename}', 'r') as file:
      documents.append(file.read())
      metadatas.append({'filename': filename})

  # Get or create a Chroma collection
  collection = get_or_create_collection(collection_name=collection_name,
                                        persist_directory=persist_directory)

  # Create ids from the current count
  count = collection.count()
  print(f'Collection contains {count} documents')
  ids = [str(i) for i in range(count, count + len(documents))]

  collection.add(ids=ids, documents=documents, metadatas=metadatas)

  new_count = collection.count()
  print(f'Added {new_count - count} documents')


if __name__ == "__main__":
  # Read the data directory, collection name, and persist directory
  parser = argparse.ArgumentParser(
    description='Embed data from a directory into a Chroma collection')

  # Add arguments
  parser.add_argument('--data_directory',
                      type=str,
                      help='The directory where your text files are stored')
  parser.add_argument(
    '--persist_directory',
    type=str,
    help='The directory where you want to store the Chroma collection')
  parser.add_argument('--collection_name',
                      type=str,
                      help='The name of the Chroma collection')

  # Parse arguments
  args = parser.parse_args()

  main(data_dir=args.data_directory,
       collection_name=args.collection_name,
       persist_directory=args.persist_directory)
