from flask import Blueprint, request, jsonify
from flask_login import login_required
from flask_cors import cross_origin
import pandas as pd
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

open_AI_routes = Blueprint('open_AI_routes', __name__)
client = OpenAI(api_key= os.environ['your_api_key'])

def get_response(chat):
    # Create a chat completion with the conversation history
    #Start-time utilizing decorator
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=chat
    )#
    #Endtime
    # Return the response text
    return completion.choices[0].message.content

def find_and_save_model_files(base_path):
    model_directories = ['model', 'models', 'entity']
    all_file_contents = []
    allowed_extensions = ('.java', '.class', '.py', '.js')
    counter = 0

    for root, dirs, files in os.walk(base_path):
        if any(model_dir in root for model_dir in model_directories):
            for file in files:
                if file.endswith(allowed_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            file_contents = file.read()
                            all_file_contents.append({'file_path': file_path, 'content': file_contents})
                            counter+=1
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    print(counter)
    return all_file_contents

def handle_initial_schema(base_path):
    model_files = find_and_save_model_files(base_path) #Finds the file contents
    user_input = f"Project model and backend contents:\n{model_files}\n\nThe above lines are the model contents, create a schema from the contents. What questions do you have regarding your schema?"
    
    # This message should be a simple string, not a formatted response
    return [{"role": "user", "content": user_input}]

@open_AI_routes.route('/ask-openai', methods=['POST'])
def ask_openai():
    try:
        data = request.get_json()

        if 'user_input' not in data:
            return jsonify({'error': 'Missing user_input parameter'}), 400

        user_input = data['user_input']
        chat = data.get('chat', [])

        if 'base_path' in data and not chat:
            base_path = data['base_path']
            chat.extend(handle_initial_schema(base_path))

        # Add new user message in the correct format
        chat.append({"role": "user", "content": user_input})

        response = get_response(chat)

        # Ensure get_response returns only the assistant's response text
        chat.append({"role": "assistant", "content": response})

        return jsonify({'response': response, 'chat': chat})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500