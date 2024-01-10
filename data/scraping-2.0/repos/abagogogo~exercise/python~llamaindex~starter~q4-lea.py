import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
)
from llama_index.llms import OpenAI
import logging
import openai
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

openai.api_key = os.environ['OPENAI_API_KEY']
llm = OpenAI(model="gpt-4", tempeature=0)
service_context = ServiceContext.from_defaults(llm=llm)

# check if storage already exists
PERSIST_DIR = "./storage-lea"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data-lea").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

#query_engine = index.as_query_engine()
query_engine = index.as_query_engine(service_context=service_context, similarity_top_k=3)
#response = query_engine.query("Brief TMAP profile") # OK -> Good
#print(response)

while True:
  q = input("Question : ")
  if q:
    response = query_engine.query(q)
    print("=" * 80 + "\n") 
    print("User input:\n\"%s\"\n" % q) 
    print("Answer:\n%s\n" % response)
