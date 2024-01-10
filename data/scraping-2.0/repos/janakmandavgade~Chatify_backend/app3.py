from flask import Flask, request, render_template, session, redirect, url_for
from flask_uploads import UploadSet, configure_uploads, DOCUMENTS
import PyPDF2
import openai
import os
from dotenv import load_dotenv
import sys
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
import textwrap
from langchain.chat_models import ChatOpenAI
import chromadb
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from sqlalchemy import Column, Integer, String
# from sqlalchemy.ext.declarative import declarative_base
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, engine
from sqlalchemy.orm import declarative_base, sessionmaker
from transformers import pipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from flask_pymongo import PyMongo
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from flask import jsonify
from bson.json_util import dumps
import uuid
from flask_cors import CORS, cross_origin
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from datetime import datetime, timedelta
from functools import wraps
from firebase_admin import credentials, initialize_app, storage
import config

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
load_dotenv()


# app.secret_key = os.environ.get('SECRET_KEY')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')

# Flask-Uploads
pdfs = UploadSet('documents', DOCUMENTS)



app.config['MONGO_URI'] = os.environ.get('MONGO_URI')
app.config['UPLOADED_DOCUMENTS_DEST'] = 'static/files'

configure_uploads(app, pdfs)

# Firebase
# cred = credentials.Certificate("firebasekey.json")
# initialize_app(cred, {
#     'storageBucket': 'gs://chatifyai-381ff.appspot.com'
# })
# firebase_admin.initialize_app(cred)

mongo_uri = os.environ.get("MONGO_URI")
mongo = MongoClient(
    mongo_uri)

openai.api_key = os.environ.get("OPENAI_API_KEY")


# IMPLEMENTED JWT AUTHENTICATION

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']
            print("token from frontend:", token)
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            print("reached inside try:", token)
            data = jwt.decode(
                token, "secret", algorithms=["HS256"])
            console.log("reached inside try")
            current_user = mongo.db.users.find_one(
                {'public_id': data['public_id']})
        except:
            return jsonify({'message': 'Token is invalid!', 'token': token}), 401
        return f(current_user, *args, **kwargs)
    return decorated


@app.route('/signup/', methods=['POST'])
@cross_origin()
def signup():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password are required!'}), 400

    existing_user = mongo.db.users.find_one({'email': email})

    if existing_user:
        return jsonify({'message': 'User already exists!'}), 400

    hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

    user_id = mongo.db.users.insert({
        'public_id': str(uuid.uuid4()),
        'email': email,
        'password': hashed_password
    })

    return jsonify({'public_id': str(user_id)}), 201


@app.route('/login/', methods=['POST'])
@cross_origin()
def login():
    data = request.get_json()

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password are required!'}), 400

    user = mongo.db.users.find_one({'email': email})

    if not user:
        return jsonify({'message': 'User not found!'}), 404

    if check_password_hash(user['password'], password):
        token = jwt.encode({
            'public_id': user['public_id'],
            'exp': datetime.utcnow() + timedelta(minutes=30)
        }, app.config['SECRET_KEY'], algorithm="HS256")
        print("token from backend:", token)
        return jsonify({'token': token, 'public_id': user['public_id']}), 200
    else:
        return jsonify({'message': 'Wrong password!'}), 401


@token_required
@app.route('/upload/<chat_session_id>/', methods=['POST'])
@cross_origin()
def upload(chat_session_id):
    if request.method == 'POST' and 'pdf' in request.files:
        print('started')
        filename = pdfs.save(request.files['pdf'])
        # session['filename'] = filename
        print(filename)
        mongo.db.question_responses.update_one(
            {"chat_session_id": chat_session_id},
            {"$push": {"documents": filename}},
            upsert=True
        )

        return jsonify({"filename": filename, "chat_session_id": chat_session_id})
    return jsonify({"message": "Upload a file."})


@app.route('/curruser/', methods=['GET'])
@cross_origin()
def me(current_user):
    return jsonify({'public_id': current_user['public_id']}), 200


@token_required
@app.route('/ask/<chat_session_id>/', methods=['POST'])
@cross_origin()
def ask(chat_session_id):

    if request.method == 'POST':
        question = request.form['question']
        chat_session = mongo.db.question_responses.find_one(
            {"chat_session_id": chat_session_id})
        if not chat_session or not "documents" in chat_session or not chat_session["documents"]:
            return jsonify({"answer": "No file uploaded. Please upload a file."})
        filename = chat_session["documents"][-1]

        pdf_path = os.path.join('static\\files', filename)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

# NEW CODE

        def split_docs(documents, chunk_size=1000, chunk_overlap=200):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            docs = text_splitter.split_documents(documents)
            return docs

        docs = split_docs(documents)

        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2")
        c2db = Chroma.from_documents(docs, embeddings)

        persist_directory = "chroma_db"

        vectordb = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persist_directory)

        vectordb.persist()

        new_db = Chroma(persist_directory=persist_directory,
                        embedding_function=embeddings)

        model_name = "gpt-3.5-turbo"
        llm = ChatOpenAI(model_name=model_name)

        chain = load_qa_chain(llm, chain_type="stuff")
        matching_docs = c2db.similarity_search(question)
        excerpts = matching_docs[0].page_content
        answer = chain.run(input_documents=matching_docs, question=question)

        mongo.db.question_responses.update_one(
            {"chat_session_id": chat_session_id},
            {"$push": {"messages": {"question": question, "answer": answer}}}
        )

        return jsonify({"question": question, "answer": answer, "excerpts": excerpts})
    else:
        return jsonify({"message": "Ask a question."})

# @token_required
# @cross_origin()


@app.route('/clear', methods=['GET'])
def clear():
    mongo.db.question_responses.delete_many({})
    mongo.db.chat_sessions.delete_many({})
    mongo.db.users.delete_many({})
    return jsonify({"message": "Database cleared!"})


@app.route('/new_chat/', methods=['POST'])
@cross_origin()
def new_chat():
    # Create a new chat session
    user_id = request.headers.get('user_id')
    print("User ID: ", user_id)
    chat_session_id = str(uuid.uuid4())
    mongo.db.question_responses.insert_one(
        {"chat_session_id": chat_session_id, "user_id": user_id, "messages": []})
    return jsonify({"message": "New chat session created.", "chat_session_id": chat_session_id})


@token_required
@cross_origin()
@app.route('/get_chat_session_id/', methods=['GET'])  # verified
def get_chat_session_id():
    userID = request.headers.get('userID')
    # chat_session = mongo.db.question_responses.find_one(sort=[("_id", -1)])
    chat_session = mongo.db.question_responses.find_one(
        {"user_id": userID}, sort=[("_id", -1)])
    if chat_session:
        return jsonify({"chat_session_id": chat_session["chat_session_id"]})
    else:
        return jsonify({"message": "No active chat session."})


@token_required
@app.route('/get_chats/<chat_session_id>/', methods=['GET'])
@cross_origin()
def get_chats(chat_session_id):
    userID = request.headers.get('userID')
    # chat_session = mongo.db.question_responses.find_one(
    # {"chat_session_id": chat_session_id})

    chat_session = mongo.db.question_responses.find_one(
        {"chat_session_id": chat_session_id, "user_id": userID})

    if chat_session:
        return dumps(chat_session["messages"])
    else:
        return jsonify({"message": "Chat session not found."})


@token_required
@app.route('/get_all_chat_sessions/', methods=['GET'])  # verified
@cross_origin()
def get_all_chat_sessions():
    userID = request.headers.get('userID')
    print("User ID: ", userID)
    # chat_sessions = mongo.db.question_responses.find()
    chat_sessions = mongo.db.question_responses.find({"user_id": userID})
    return dumps(chat_sessions)
