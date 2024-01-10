from llama_index import VectorStoreIndex, SimpleDirectoryReader
import os
import openai
import subprocess
#gets api key from environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_embeddings(path, storage_path)):
    subprocess.call(['mkdir', 'temp.data',])
    subprocess.call(['cp', path, 'temp.data',])
    documents = SimpleDirectoryReader('temp.data').load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(storage_path)
    subprocess.call(['rm', '-rf', 'temp.data',])

    return "Embeddings created! "


