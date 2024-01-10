# sqlite fix for python 3.10.2
# https://docs.trychroma.com/troubleshooting#sqlite
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# langchain imports
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import Pinecone
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import TokenTextSplitter

# other imports
import pinecone
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from pprint import pprint

load_dotenv()


################################################
# load documents
################################################

# load solidity smart contracts
loader = DirectoryLoader('contracts', glob="**/*.sol", show_progress=True) #silent_errors=True

sol_docs = loader.load()

print(f"Number of Solidity contracts loaded: {len(sol_docs)}")

# load json files
# json_loader = JSONLoader(
#    file_path='contracts/vulns.jsonl',
#    jq_schema='.body',
#    text_content=False,
#    json_lines=True)

# json_docs = json_loader.load()

# full array of all documents
# sol_docs.extend(json_docs)

docs = sol_docs

# print(f"Number of total documents loaded: {len(docs)}")

################################################
#  split documents into chunks
################################################

text_splitter = TokenTextSplitter(chunk_size=1024, chunk_overlap=0)

doc_chunks = []

for doc_index, doc in enumerate(docs):
    print(f"Processing document {doc_index + 1}/{len(docs)}")
    sol_code = doc.page_content
    
    split_docs = text_splitter.split_text(sol_code)

    for chunk in split_docs:
        #doc_chunks.append(chunk)
        doc_chunks.append(Document(page_content=chunk, metadata={}))

print(f"Number of documents after splitting: {len(doc_chunks)}")

################################################
#  create embeddings
################################################
print("Creating embeddings...")

# embedding function
embeddings_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-ada-002")

# embeddings = embeddings_model.embed_documents(doc_chunks)

################################################
#  create vector db (embeddings performed by function)
################################################
print("Initiating vector database...")

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV"),  # next to api key in console
)

index_name = "auditme"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
# The OpenAI embedding model `text-embedding-ada-002 uses 1536 dimensions`
docsearch = Pinecone.from_documents(doc_chunks, embeddings_model, index_name=index_name)

# if you already have an index, you can load it like this
# docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Create chromadb vector store
# db = Chroma.from_documents(doc_chunks, OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-ada-002"))

