from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from openai import OpenAI
import os

# Flask app setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# Environment Variables for Sensitive Data
mongo_uri = os.environ.get('MONGO_URI')
api_key = os.environ.get('OPENAI_API_KEY')

# MongoDB setup
client_mongo = MongoClient(mongo_uri)

# OpenAI client setup
client_openai = OpenAI(api_key=api_key)

# MongoDB collection
db = client_mongo.veda_chat_db
chat_history_collection = db.chat_history

@app.route('/')
def index():
    return "Welcome to Veda Chat!"

@app.route('/chat', methods=['POST'])
def chat():
    session_id = request.json.get('session_id')
    user_input = request.json.get('input')
    try:
        response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            temperature=1,
            max_tokens=4000,
            top_p=0.5,
            frequency_penalty=1,
            presence_penalty=0.25
        )
        chat_response = response.choices[0].message['content']

        # Store in MongoDB Atlas
        chat_history_collection.insert_one({'session_id': session_id, 'input': user_input, 'response': chat_response})
        return jsonify({'response': chat_response})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/history', methods=['GET'])
def history():
    session_id = request.json.get('session_id')
    all_chats = list(chat_history_collection.find({'session_id': session_id}, {'_id': 0}))
    return jsonify(all_chats)
