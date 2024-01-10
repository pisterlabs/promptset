from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

#Setting Environment variables
from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')



# app instance
app = Flask(__name__)
CORS(app)
@cross_origin()
@app.route("/api/home", methods=['POST'])
def chat_document():
    data = request.get_json()
    pdfUrl = data['url']
    query = data['chat']

    #Load PDF
    #The url should be coming from the front end through a post request
    loader = PyPDFLoader(pdfUrl)
    if loader:
        data = loader.load_and_split()
    else:
        return "Error loading PDF"

    #Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)

    #Embedding and vector storage
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(texts, embeddings)

    #query
    # query = "What's the main point of the document?"
    docs = vectorstore.similarity_search(query)

    #Load LLM and chatchain
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    llmresponse = chain.run(input_documents=docs, question=query)

    response = jsonify({
        'message': llmresponse,
        'role': 'ai'
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

# if __name__ == "__main__":
#     app.run(debug=True, port=8080)

# app = Flask(__name__)
# CORS(app)
# @cross_origin()
@app.route("/api/guest", methods=['POST'])
def guest_document():
    data = request.get_json()
    pdfUrl = data['url']
    query1 = data['chat1']
    query2 = data['chat2']

    #Load PDF
    #The url should be coming from the front end through a post request
    loader = PyPDFLoader(pdfUrl)
    if loader:
        data = loader.load_and_split()
    else:
        return "Error loading PDF"

    #Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(data)

    #Embedding and vector storage
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(texts, embeddings)

    #query
    # query = "What's the main point of the document?"
    docs1 = vectorstore.similarity_search(query1)
    docs2 = vectorstore.similarity_search(query2)

    #Load LLM and chatchain
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    llmresponse1 = chain.run(input_documents=docs1, question=query1)
    llmresponse2 = chain.run(input_documents=docs2, question=query2)
    response = jsonify({
        'message1': llmresponse1,
        'message2': llmresponse2,
        'role': 'ai'
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response

if __name__ == "__main__":
    app.run(debug=True, port=8080)

#Load PDF
#The url should be coming from the front end through a post request
# loader = PyPDFLoader("https://cloud.appwrite.io/v1/storage/buckets/64e828dda98159be482f/files/32542b6a-bc17-40de-b846-a959f0e42861/view?project=64e823bf4acf38b1d573&mode=admin")
# data = loader.load_and_split()

# #Text Splitting
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# texts = text_splitter.split_documents(data)

# #Embedding and vector storage
# embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# vectorstore = FAISS.from_documents(texts, embeddings)

# #query
# query = "What's the main point of the document?"
# docs = vectorstore.similarity_search(query)

# #Load LLM and chatchain
# llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
# chain = load_qa_chain(llm, chain_type="stuff")
# response = chain.run(input_documents=docs, question=query)

# print("Successfully ran llmpython.py:", response)