from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from llama_index.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# variables
llm_model = "mistral:latest"
ollama_host = "http://localhost:11434"
PERSIST_DIR = "./storage"

llm = Ollama(model=llm_model)
embed_model = OllamaEmbeddings(base_url=ollama_host, model=llm_model)

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("schedule").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist()
else:
    # TODO: set the right use of service context
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR, service_context=service_context)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()
response = query_engine.query("What is the data about answer in one word?")
print(response)
