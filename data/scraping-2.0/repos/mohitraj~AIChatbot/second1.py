from llama_index import StorageContext, load_index_from_storage
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
import os
import openai

from key1 import KEY 
import os
import openai

os.environ['OPENAI_API_KEY'] = KEY 
	


def generate_response(prompt):
	storage_context = StorageContext.from_defaults(persist_dir="./vectordatabase")
	index = load_index_from_storage(storage_context)
	query_engin = index.as_query_engine() 
	question = prompt
	response = query_engin.query(question)
	return str(response)
	#print ("\n", response)

