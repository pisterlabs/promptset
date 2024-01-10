from pymongo import MongoClient
from dotenv import load_dotenv
import os
import openai
import json
from flask import request, jsonify, make_response

#Load environment variables from .env file
load_dotenv() 

openai.api_key = os.getenv("OPENAI_API_KEY")

client = MongoClient(os.getenv("MONGO_URL"))
db = client["app"]
collection = db["voicebot_status"]

# voice bot functions
#check if the response is successful

def get_doc(document):
    if document:
        document["_id"] = str(document["_id"])
        return document
    else:
        return None

def check_status(user_id):

    try:
        # Retrieve the status of the process
        status = collection.find_one({'user_id': user_id})

        if status is None:
            return make_response(jsonify({"message" : "not found"}), 404)
        
        return make_response(jsonify({"status" : get_doc(status)}), 200)
    except Exception as e:

        return make_response(jsonify({"message": "server error"}), 500)

#initializing the mongoDB document for the request
def initialize_status(user_id):
    try:
        initial_status = { "user_id": user_id,"status": "started"}

        res = collection.find_one({"user_id": user_id})

        if res is None:
            result = collection.insert_one(initial_status)
        else:
            result = collection.replace_one({"user_id": user_id},initial_status, upsert=True)
        
        return str(result["_id"])
    except Exception:
        
        return make_response(jsonify({"message": "server error"}), 500)

#take audio file and transcribe into text format. To be edited further
def take_prompt():

    try:
        language_codes = ["en"]
        audio_file = request.files['audio']
        audio_file.save(audio_file.filename)
        with open(audio_file.filename, 'rb') as f:
            transcript = openai.Audio.transcribe("whisper-1", f, models=language_codes, response_format="text")

    except openai.error.RateLimitError:
        transcript = None
        filter_query = {"user_id": request.user_id}
        update_query = {"$set": {"status": "error", "error": "Rate limit exceeded. Try again after few seconds."}}
        collection.update_one(filter_query, update_query)

        return make_response(jsonify({"message": "Rate limit error."}), 429)
    
    except openai.error.AuthenticationError:
        transcript = None
        filter_query = {"user_id": request.user_id}
        update_query = {"$set": {"status": "error", "error": "error in transcription"}}
        collection.update_one(filter_query, update_query)

        return make_response(jsonify({"message": "Authentication error. Check OPEN AI API key."}), 401)
    
    except Exception:
        
        return make_response(jsonify({"message": "server error"}), 500)

    #update status of in mongoDB
    filter_query = {"user_id": request.user_id}
    update_query = {"$set": {"status": "transcribed","prompt": transcript}}
    collection.update_one(filter_query, update_query)
        
    return transcript


#execute the command received from the prompt
def execute_command(prompt):

    print("prompt",prompt)
    
    #get the commands to provide the prompt from the apis.json file
    try:
        file = open("./files/apis.json", "r")
        command_id = []
        command_name = []
        command_description = []

        api_format = json.load(file)
        

        for x in api_format:
            command_id.append(x['id'])
            command_name.append(x['name'])
            command_description.append(x['description'])

    except Exception:
        
        filter_query = {"user_id": request.user_id}
        update_query = {"$set": {"status": "error", "error": "error in finding apis.json"}}
        collection.update_one(filter_query, update_query)
       
        return make_response(jsonify({"message": "api file not found"}), 404)
        
        
    #user prompt to identify the command    
    user_prompt = f'categorize prompt: {prompt}. id {command_id}. name {command_name}. description {command_description}.'
    print(user_prompt)
    #provide the prompt to the openai api
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [{"role": "system", "content": "you are a human child trying to categorize the received user prompts into given categories. give only single id as output without extra space. No other explanation."},
        {"role": "user", "content": user_prompt}],
        #temperature=0.5,
        max_tokens=9,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0
    )

    try:
        #receive the response from the openai api and convert it to a dictionary
            #response will be in an integer
        response = response['choices'][0]['message']['content']
        id = int(response)
        print("id :", id)
    except ValueError:
        id = 5
        filter_query = {"user_id": request.user_id}
        update_query = {"$set": {"status": "error", "error": "Couldn't get proper prompt. Try again!"}}
        collection.update_one(filter_query, update_query)
        return make_response(jsonify({"message" : "Couldn't get proper prompt. Try again!"}))
    
    except openai.error.RateLimitError:

        filter_query = {"user_id": request.user_id}
        update_query = {"$set": {"status": "error", "error": "Rate limit exceeded. Try again after few seconds"}}
        collection.update_one(filter_query, update_query)
        return make_response(jsonify({"message": "Rate limit error. Try again after few seconds."}), 429)
    
    except openai.error.AuthenticationError:

        filter_query = {"user_id": request.user_id}
        update_query = {"$set": {"status": "error", "error": "error in extracting api details"}}
        collection.update_one(filter_query, update_query)

        return make_response(jsonify({"message": "Authentication error. Check OPEN AI API key."}), 401)
    
    except Exception:
        
        return make_response(jsonify({"message": "server error"}), 500)
    
    try:   
    #get the format of the api call from the apis.json file
        user_prompt = f'user prompt: {prompt}.\n Required format: {api_format[id]}. fill and update the null values only. Dont change anything else.'
        #provide the prompt to the openai api
        print("2nd",user_prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= [{"role": "system", "content": "You are an examinee replacing only the null values. only format specified as output. No explanation."},
            {"role": "user", "content": user_prompt}],
            temperature=0.5,
            #max_tokens=9,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0
        )
        response = response['choices'][0]['message']['content']
        filter_query = {"user_id": request.user_id}
        update_query = {"$set": {"status": "processed"}}
        collection.update_one(filter_query, update_query)
    
    
        if eval(response):

            response = eval(response)
    
    except SyntaxError:
            
            response = api_format[5]
            filter_query = {"user_id": request.user_id}
            update_query = {"$set": {"status": "error", "error": "Couldn't get proper prompt. Try again!"}}
            collection.update_one(filter_query, update_query)
            return make_response(jsonify({"message" : "Couldn't get proper prompt. Try again!"}))

    except openai.error.RateLimitError:

        filter_query = {"user_id": request.user_id}
        update_query = {"$set": {"status": "error", "error": "Rate limit exceeded. Try again after few seconds"}}
        collection.update_one(filter_query, update_query)
        return make_response(jsonify({"message": "Rate limit error. Try again after few seconds."}), 429)
    
    except openai.error.AuthenticationError:

        filter_query = {"user_id": request.user_id}
        update_query = {"$set": {"status": "error", "error": "error in extracting formatted response"}}
        collection.update_one(filter_query, update_query)

        return make_response(jsonify({"message": "Authentication error. Check OPEN AI API key."}), 401)
    
    except Exception:
        
        return make_response(jsonify({"message": "server error"}), 500)
    
    filter_query = {"user_id": request.user_id}
    update_query = {"$set": {"status": "completed", "success": response}}
    collection.update_one(filter_query, update_query)

    return make_response(jsonify({"message": "success"}), 200)
