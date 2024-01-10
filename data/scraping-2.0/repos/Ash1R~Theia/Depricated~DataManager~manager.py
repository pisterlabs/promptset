from flask import Flask, request
import csv
import requests
import csv 
import json
import time
from flask_socketio import SocketIO, emit
from os.path import exists
import subprocess

import os
import logging
from flask import jsonify
from flask_executor import Executor
import openai


import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your OpenAI API key
openai.api_key = "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"


def csv_to_json(csvFilePath, jsonFilePath):
    jsonArray = []
      
    #read csv file
    with open(csvFilePath, encoding='utf-8') as csvf: 
        #load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            jsonArray.append(row)
  
    #convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf: 
        jsonString = json.dumps(jsonArray, indent=4)
        jsonf.write(jsonString)

def post_json(url, data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    
    # Check the response status code
    if response.status_code == 200:
        print("POST request successful!")
    else:
        print("POST request failed.")
    
    # Print the response content
    print(response.json())

def generate_prompt(user_message):
    # Analyze the user_message and generate a suitable system message.
    if 'math' in user_message.lower():
        return "You are a helpful tutor specializing in Math."
    elif 'science' in user_message.lower():
        return "You are a helpful tutor specializing in Science."
    elif 'english' in user_message.lower():
        return "You are a helpful tutor specializing in English."
    else:
        return "You are a helpful tutor."

def generate_response(msg):
    try:
        system_message = generate_prompt(msg)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg}
            ]
        )
        # Return the AI's response as JSON
        return jsonify({'message': response['choices'][0]['message']['content']})
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({'message': "Sorry, I'm having trouble generating a response right now."})

def append_to_json_file(filepath, data):
    try:
        with open(filepath, "r+") as file:
            file_data = json.load(file)
            file_data.append(data)
            file.seek(0)
            json.dump(file_data, file)
    except FileNotFoundError:
        with open(filepath, "w") as file:
            json.dump([data], file)


# Initialize the Firebase Admin SDK
cred = credentials.Certificate("DataManager/serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'theia-b3690.appspot.com'
})

# Get a reference to the storage bucket
bucket = storage.bucket()

app = Flask(__name__)
socketio = SocketIO(app)

fields = ["user", "message"]

classId = ""
user = ""
message = ""
@app.route('/receive', methods=['POST'])
def receive_data():
    data = request.get_json()  # Extract the JSON data from the request
    classId = data.get('class')
    print(classId)
    user = data.get('user')
    print(user)
    message = data.get('message')
    print(message)
    try:
        user_message = request.json['message']
        logging.info('Received message: ' + user_message)
        if user_message.startswith("/gpt"):
            # Extract the question for GPT
            gpt_question = user_message[5:]  # Skip the "/gpt " part
            response = generate_response(gpt_question)
            # Append the question and response to the chat log

            append_to_json_file("DataManager/Data/class" + classId +".csv", {"question": gpt_question, "response": response.json['message']})
            # Specify the local file path and desired remote file name
            local_file_path = "DataManager/Data/class" + classId +".csv"
            remote_file_name = "class" + classId+".json"

            blob = bucket.blob(remote_file_name)
            blob.upload_from_filename(local_file_path)

            print("File uploaded successfully.")
            return response
        else:
            # If it's not a /gpt command, just return a generic response
            return jsonify({'message': "Command not recognized. Please start your command with '/gpt'."})
    except:
        # Process the data and display it
        if data:
            # Assuming the JSON data has a key called 'message'
            classId = data.get('class')
            print(classId)
            user = data.get('user')
            print(user)
            message = data.get('message')
            print(message)
            listToAppend = [user, message]
            print(listToAppend)
            if exists("DataManager/Data/class" + classId +".csv"):
                w = csv.writer(open(r"DataManager/Data/class" + classId +".csv", 'a', newline='', encoding='UTF8'), dialect='excel')
                w.writerow(listToAppend)
            else:
                with open(r"DataManager/Data/class" + classId +".csv", 'w', newline='', encoding='UTF8') as f:
                    # create the csv writer
                    writer = csv.writer(f)
                    writer.writerow(fields)
                    writer.writerow(listToAppend)

            csvFilePath = r"DataManager/Data/class" + classId +".csv"
            jsonFilePath = r"DataManager/Data/class" + classId +".json"

            csv_to_json(csvFilePath, jsonFilePath)

            # Specify the local file path and desired remote file name
            local_file_path = "DataManager/Data/class" + classId +".csv"
            remote_file_name = "class" + classId+".json"

            blob = bucket.blob(remote_file_name)
            blob.upload_from_filename(local_file_path)

            print("File uploaded successfully.")
            return "Invalid data"

@app.route('/')
def index():
    return "Chat Interface"



if __name__ == '__main__':
    socketio.run(app,host='0.0.0.0', port=5000, debug=True)