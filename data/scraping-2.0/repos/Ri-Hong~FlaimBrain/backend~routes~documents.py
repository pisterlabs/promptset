from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
from bson import ObjectId
from db import get_db  # Ensure you have this helper function to get a db instance
from werkzeug.utils import secure_filename
import os

from datetime import datetime
from pymongo import MongoClient
from langchain.document_loaders import PyMuPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

documents = Blueprint('documents', __name__)


# Creates an embedding, stores it in Chroma, and returns the text content of the document
def upload_data_to_vector_db(file, persist_directory):
    if not file or file.filename == '':
        return ""

        # return "No selected file", 400
    # Save the file to a temporary location
    filename = secure_filename(file.filename)
    temp_file_path = os.path.join("/tmp", filename)
    file.save(temp_file_path)
    
    print("TFP: ", temp_file_path)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension == '.pdf':
            loader = PyMuPDFLoader(temp_file_path)
        elif file_extension == '.txt' or file_extension == '.png':
            loader = TextLoader(temp_file_path)
        else:
            return jsonify({"message": "Invalid file type"}), 400
        documents = loader.load()

        # Split the document text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=0,
            separators=["\n\n", "\n", "(?<=\. )", " ", ""]
        )
        
        texts = text_splitter.split_documents(documents)

        print("Texts: ", texts)

        # Generate embeddings
        embeddings = OpenAIEmbeddings()

        # Store vectors in Chroma
        vectordb = Chroma.from_documents(documents=texts, 
                                        embedding=embeddings,
                                        persist_directory=persist_directory)
        vectordb.persist()


        document = documents[0]
        text_content = document.page_content
    finally:
        os.remove(temp_file_path)  # Delete the temporary file
    print(f"Embeddings for {filename} added to the vector database.")

    return text_content


@documents.route('/create', methods=['POST'])
@jwt_required()
def create_document():
    print("Creating document")
    db = get_db()
    user_id = get_jwt_identity()  # Get the identity of the current user


    file = request.files['file'] if 'file' in request.files else None
    fileName = request.form['fileName']
    file_or_folder = request.form['fileOrFolder']
    parent_id = request.form['parentId'] if 'parentId' in request.form else None

    page_content = upload_data_to_vector_db(file, "./storage")

    new_document = {
        "userId": user_id,
        "name": fileName,
        "type": file_or_folder,
        "parentId": parent_id,
        "children": [],
        "content": page_content,
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    }
    
    doc = db.documents.insert_one(new_document)

    return jsonify({"message": "Document created", "id": str(doc.inserted_id)}), 201

@documents.route('/get', methods=['GET'])
@jwt_required()
def get_documents():
    db = get_db()
    user_id = get_jwt_identity()
    user_documents = list(db.documents.find({"userId": user_id}))
    
    # Convert ObjectId instances to strings for JSON serialization
    for doc in user_documents:
        doc["_id"] = str(doc["_id"])
    
    return jsonify(user_documents)

@documents.route('/delete/<doc_id>', methods=['DELETE'])
@jwt_required()
def delete_document(doc_id):
    db = get_db()
    user_id = get_jwt_identity()  # Get the identity of the current user

    # Find the document to ensure it belongs to the current user
    document = db.documents.find_one({"_id": ObjectId(doc_id), "userId": user_id})

    if document:
        # Delete the document
        db.documents.delete_one({"_id": ObjectId(doc_id)})
        return jsonify({"message": "Document deleted"}), 200
    else:
        return jsonify({"message": "Document not found"}), 404

