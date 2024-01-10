import os
import openai
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv('.env')

# Initialize Flask app
app = Flask(__name__)

# Load OpenAI embeddings
def load_embeddings():
    client = openai.api_key = os.getenv('OPENAI_API_KEY') # Use environment variable
    embeddings = OpenAIEmbeddings(model="ada", client=client)
    return embeddings

# Initialize Pinecone index
def init_index(embeddings):
    pinecone.init(api_key=os.environ.get('PINECONE_API'), environment="asia-northeast1-gcp")
    index_name = "qabot2"
    index = pinecone.Index(index_name, embeddings)
    return index

# Load language model
def load_language_model():
    client = os.environ.get('OPENAI_API_KEY')  # Use environment variable
    llm = OpenAI(model="text-davinci-003", client=client)
    return llm

# Initialize question-answering chain
def initialize_qa_chain(llm):
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

# Load language model
llm = load_language_model()

# Initialize question-answering chain
chain = initialize_qa_chain(llm)

# Get similar documents based on a query
def get_similar_docs(query, k=1, score=False):
    if score:
        similar_docs = index.similarity_search_with_score(query, k=k)
    else:
        similar_docs = index.similarity_search(query, k=k)
    return similar_docs

# Get the answer to a question
def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

# Define the Cloud Function endpoint
@app.route('/', methods=['POST'])
def cloud_function(request):
    # Get the query from the request data
    data = request.get_json()
    query = data['query']

    # Call the get_answer function
    answer = get_answer(query)

    # Return the answer as a JSON response
    return jsonify({'answer': answer})

# Load OpenAI embeddings
embeddings = load_embeddings()

# Initialize Pinecone index
index = init_index(embeddings)

# Entry point for the Cloud Function
def main(request):
    return app(request)
