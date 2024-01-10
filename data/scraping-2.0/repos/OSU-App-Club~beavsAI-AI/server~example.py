# © 2023 App Development Club @ Oregon State Unviersity
# All Rights Reserved

# This script is used to load "arbitrary" documents into the vector store and run a query against them

import os

import pinecone
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

load_dotenv()

# You'll need to get your own API keys from Pinecone and OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_API_ENV = os.environ["PINECONE_API_ENV"]

print(f"\n Loading PDF... \n")
# Change this to the path of the PDF you want to load
loader = UnstructuredPDFLoader("../data/syllabus/CS162_F23.pdf")

# Load the PDF
data = loader.load()

print(f"\n Splitting PDF... \n")

# Chunk the data into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print(f"\nSuccesssfully Split. You now have {len(texts)} documents \n")

print(f"\n Loading Embeddings... \n")

# Create embeddings of the documents to get ready for semantic search
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at https://www.pinecone.io/
    environment=PINECONE_API_ENV,  # find next to the API key
)

# Change this to the name of the index you want to use (or create)
index_name = "cs162-index"

# Create the index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

print(f"\n Loading Documents... \n")

# Load the documents into the vector store
docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# Run a similarity search
query = "What are the pre-requisites for this course?"
docs = docsearch.similarity_search(query)

print(f"\n Running Question Answering Chain... \n")

# Load the question answering chain
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

# Run the question answering chain
llm_response = chain.run(input_documents=docs, question=query)

print(f"\n Question Answering Chain Response: \n")
print(llm_response)