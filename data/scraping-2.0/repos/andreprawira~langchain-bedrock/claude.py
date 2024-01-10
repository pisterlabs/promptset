import boto3
import json
import time
import os
from dotenv import load_dotenv

from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import load_qa_chain

import pinecone

load_dotenv()

bedrock_runtime = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1"
)

modelId = 'anthropic.claude-v2'
accept = 'application/json'
contentType = 'application/json'
body = json.dumps({
    "max_tokens_to_sample": 40000,
    "temperature": 0.1,
    "top_p": 0.9,
})

###
# Define the path to the directory containing the PDF files (example_data folder).
directory = 'data'

# Function to load documents from the specified directory.
def load_docs(directory):
    # Create an instance of the DirectoryLoader with the provided directory path.
    loader = DirectoryLoader(directory)
    # Use the loader to load the documents from the directory and store them in 'documents'.
    documents = loader.load()
    # Return the loaded documents.
    return documents


# Call the load_docs function to retrieve the documents from the specified directory.
documents = load_docs(directory)

# Function to split the loaded documents into semantically separate chunks.
def split_docs(documents, chunk_size=256, chunk_overlap=25):
    # Create an instance of the RecursiveCharacterTextSplitter with specified chunk size and overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # Use the text splitter to split the documents into chunks and store them in 'docs'.
    docs = text_splitter.split_documents(documents)
    # Return the split documents.
    return docs

# Call the split_docs function to break the loaded documents into chunks.
# The chunk_size and chunk_overlap parameters can be adjusted based on specific requirements.
docs = split_docs(documents)

# Initiate the pinecone client

pinecone_api = "96696670-00df-4fa2-9aad-737d9e13887a"
pinecone_env = "gcp-starter"

pinecone.init(
    api_key = pinecone_api,
    environment = pinecone_env
)

index_name = "test" # change to your index name

llm = Bedrock(
    model_id=modelId,
    client=bedrock_runtime
)
bedrock_embeddings = BedrockEmbeddings(client=bedrock_runtime)

docsearch = Pinecone.from_texts(
    [t.page_content for t in docs],
    bedrock_embeddings,
    index_name = index_name
)

chain = load_qa_chain(llm, chain_type = "stuff")
query = "what is the default user pool region"
docs = docsearch.similarity_search(query)
print(chain.run(input_documents = docs, question = query))