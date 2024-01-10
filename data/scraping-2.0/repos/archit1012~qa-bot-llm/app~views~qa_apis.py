from flask import Flask, request, jsonify, Blueprint
from dotenv import load_dotenv
from service.qa_apis_service import process_request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

load_dotenv()

# Global variable
embeddings = OpenAIEmbeddings()
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

def hello_world():
    return 'Hello World'


def upload_files():
    try:
        # Check if the POST request has file parts
        if 'doc_file' not in request.files or 'question_file' not in request.files:
            return jsonify({'error': 'Both files must be provided'})
        response = process_request(request,embeddings,chat)
        return jsonify(response)
    except Exception as e:
        print("Internal Server Error", e)
    return jsonify("Internal Server Error")