import sys
import openai

# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

if len(sys.argv) != 2:
    print("Usage: python3 app_cmd.py 'your question'")
    sys.exit(1)

param = sys.argv[1]

from llama_index import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir='./storage')
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
response = query_engine.query(param)

print(" ")
print("==> Michael AI :)", response)
print(" ") 