from flask import Flask, jsonify,make_response,request,render_template
from flask_cors import CORS
import openai
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import time
import concurrent.futures
import pandas as pd
import os
import numpy as np
from utils import *

from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

AI_LEARNED_DOCS_DIR = 'AI_learned_docs'

app = Flask(__name__)
CORS(app)

def remove_extension(filename):
    """
    This function takes a file name with extension and returns the file name without the extension.
    """
    dot_index = filename.rfind('.')
    if dot_index == -1:
        return filename
    else:
        return filename[:dot_index]


@app.route("/", methods=["GET"])
def home():
    # return "hello world"
    # return render_template('index.html')
    return render_template('index.html')



@app.route("/game/<gameid>", methods=["GET"])
def chatSession(gameid):

    filename = 'games_and_rulebooks.json'
    availableGamesIds = []
    with open(filename, 'r') as f:
        data = json.load(f)

    selectedGame = {}

    for i in data:
        if i['id'] == gameid:
            selectedGame = {'id':i['id'],"gameName":i['gameName'],"originalFileName":i['originalFileName'],"fileNameWithJson":i['fileNameWithJson']}

    if selectedGame:
        return render_template('chat_session.html',data=selectedGame)
    else:
        return "Invalid Url"




@app.route('/api/answer', methods=['POST'])
def chatbot():
    # Get the user's input and game name from the request
    print(request.form['user_input'])
    user_input = request.form['user_input']
    game_name = request.form['game_name']
    
    # Print the user's input and game name to the console
    print(f"User input: {user_input}")
    print(f"Game name: {game_name}")

    similaritiesss = search_among_documents(user_input,searchfiles=f'{AI_LEARNED_DOCS_DIR}/{game_name}')
    
    context = ""
    for i in similaritiesss:
        context += i['chunk'] + "\n"

    print("len_context: ",len(context)/4)
    ans = getAnswerFromGPT(context=context,searchQuery=user_input)

    return jsonify({'response': ans})


@app.route("/getAnswer", methods=["POST"])
def getAnswer():
    data = request.get_json()
    fileId = data['fileId']
    filename = 'games_and_rulebooks.json'
    
    
    nameOfFileFromWhichAnswerIsToBeGiven = None
    with open(filename, 'r') as f:
        data = json.load(f)
    print("fileId",fileId)
    for i in data:
        if fileId == i['id']:
            nameOfFileFromWhichAnswerIsToBeGiven = i['fileNameWithJson']
            break
    
    if nameOfFileFromWhichAnswerIsToBeGiven is not None:
            return {"status":nameOfFileFromWhichAnswerIsToBeGiven}

   
    return {"status":"404"}



@app.route("/get_list_of_games", methods=["GET"])
def getListOfAvailableGames():
    # Define the file name
    filename = 'games_and_rulebooks.json'

    # Open the file for reading or create a new file if it doesn't exist
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:

        availableDocs = os.listdir(AI_LEARNED_DOCS_DIR)
        data = []
        
        df = pd.read_excel('Board Game Chat Database.xlsx')
        for index, row in df.iterrows():

            print("file",os.path.splitext(os.path.basename(str(row['Files']))[0]+'.json'))
            print()
            fileNameWithJson = remove_extension(str(row['Files']))+'.json'
            if fileNameWithJson in availableDocs:
                data.append({
                    'id':str(time.time()).replace('.',''),
                    'gameName':row['Board Game Name'],
                    'originalFileName':row['Files'],
                    'fileNameWithJson':fileNameWithJson,
                    })
                time.sleep(0.001)
            
        with open(filename, 'w') as f:
            json.dump(data, f)

    
    
    return jsonify(data)









if __name__ == "__main__":
    app.run(debug=True,port=5002,host='0.0.0.0')