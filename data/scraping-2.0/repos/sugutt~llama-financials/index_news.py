import config
import os
import openai

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
openai.api_key = config.OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY

documents = SimpleDirectoryReader('articles').load_data()

index = GPTVectorStoreIndex.from_documents(documents)

index.storage_context.persist()