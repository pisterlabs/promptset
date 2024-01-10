from flask import Blueprint, request, jsonify, render_template
from config.db import Connection
from models.QandA import QandA
from bson import ObjectId
import os
from dotenv import load_dotenv
import jwt
from functools import wraps
import os
import openai
import tiktoken

qanda_bp = Blueprint('qanda', __name__)

# Instantiate the Connection class
db = Connection().get_db()

# Set the collection
collection = db["qanda"]
load_dotenv()


@qanda_bp.route('/tokenhelp', methods=['POST'])
def tokenhelp():
    try:
        body = request.json
        global more
        more = body["token"]
        return jsonify({"message": "success"}), 200
    except Exception as e:
        print(e)
        return jsonify({"message": "Error"}), 500


def authentication(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = more
        if not token :
            return jsonify({"status": "error", "message": "Token is missing"}), 401
        
        try:
            data = jwt.decode(token, os.getenv(
                'JWT_SECRET'), algorithms=["HS256"])
            request.environ['influencer_id'] = data['influencer_id']
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 401
        return f(*args, **kwargs)
    return decorated



@qanda_bp.route('/qanda', methods=['POST'])
@authentication
def create_qanda():
    try:
        influencer_id = request.environ['influencer_id']
        body = request.json
        body['influencer_id'] = influencer_id
        print(influencer_id)
        new_qanda = QandA(body)
        collection.insert_one(new_qanda.__dict__)
        return jsonify({"message": "Success"}), 200
    except Exception as e:
        print(e)
        return jsonify({"message": "Error"}), 500
    
@qanda_bp.route('/qandaInfo', methods=['GET'])
@authentication
def get_qanda():
    try:
        influencer_id = request.environ['influencer_id']
        qanda = list(collection.find({"influencer_id": str(influencer_id)}))
        for q in qanda:
            q['_id'] = str(q['_id'])
        return jsonify({"message": "Success", "data": qanda}), 200
    except Exception as e:
        print(e)
        return jsonify({"message": "Error"}), 500


@qanda_bp.route('/generate', methods=['GET'])
def get_manda():
    try:
        return render_template('chat.html')
    except Exception as e:
        print(e)
        return jsonify({"message": "Error"}), 500



openai.api_key = os.getenv("OPENAI_API_KEY")

@qanda_bp.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    # Extract the message content from the request JSON
    user_message = data.get('message')
    print(user_message)


    if user_message is not None:
        # Prepare the messages list for the chat completion
        messages = [
            {"role": "user", "content": user_message}
        ]

        try:
            # Perform the chat completion using the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=messages,
                temperature=1.9,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Extract the generated message from the response
            generated_message = response['choices'][0]['message']['content']

            return jsonify({"generated_message": generated_message})

        except Exception as e:
            print(e)
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "Invalid message content. Please provide a valid message."})

