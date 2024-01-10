import os
import sys
import boto3

from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain.llms import Replicate
from langchain.vectorstores import OpenSearchVectorSearch

from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain

# Replicate API token
os.environ['REPLICATE_API_TOKEN'] = "<YOUR REPLICATE API TOKEN HERE>"

# Load and preprocess the PDF document
loader = PyPDFLoader('./instructions.pdf')
documents = loader.load()

# Split the documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Use HuggingFace embeddings for transforming text into numerical vectors
embeddings = HuggingFaceEmbeddings()

# URL to connect with OpenSearch running locally. To start OpenSearch,
# run `docker-compose up` in the current directory.
local_opensearch = "http://localhost:9200"

# URL to connect with OpenSearch running on AWS. To start OpenSearch,
# run `terraform apply` in the current directory. Then, replace the
# value below with the URL provided by the Terraform output.
amazon_opensearch = "<YOUR AMAZON OPENSEARCH SERVERLESS URL HERE>"
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, 'us-east-1',
                   'aoss', session_token=credentials.token)

# Set here which OpenSearch option you intend to use. Options are:
# `local_opensearch` and `amazon_opensearch`.
opensearch_url=local_opensearch

# Set up the OpenSearch vector database
vectordb = OpenSearchVectorSearch.from_documents(
     texts,
     embeddings,
     opensearch_url=opensearch_url,
     http_auth=awsauth,
     timeout = 300,
     connection_class = RequestsHttpConnection,     
     index_name="pdf-source-docs"
)

# Initialize Replicate Llama2 Model
llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75, "max_length": 3000}
)

# Set up the Conversational Retrieval Chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_type='similarity', search_kwargs={"k": 2}),
    return_source_documents=True
)

# Start chatting with the chatbot
chat_history = []
while True:
    query = input('Prompt: ')
    if query.lower() in ["exit", "quit", "q"]:
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))