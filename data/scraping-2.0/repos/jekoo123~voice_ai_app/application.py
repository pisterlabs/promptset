import os
import base64
import uuid
import time
import secrets
from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech
from google.cloud import storage
from difflib import SequenceMatcher
from pymongo import MongoClient
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from google.oauth2.service_account import Credentials
import openai


app = Flask(__name__)

# OpenAI API 키 설정
openai.api_key = os.environ.get("OPENAI_API_KEY")



def get_secret():

    secret_name = "jekooGoogle"
    region_name = "ap-northeast-2"

    # AWS session 생성
    session = boto3.session.Session()

    # Secret Manager client 생성
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # Secret Manager에서 시크릿 가져오기
    get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    
    if 'SecretString' in get_secret_value_response:
        secret = get_secret_value_response['SecretString']
    else:
        secret = base64.b64decode(get_secret_value_response['SecretBinary'])
        
    return secret

# Google Cloud 인증 수행
def google_auth(secret):
    key_info = json.loads(secret)
    credentials = Credentials.from_service_account_info(key_info)
    return credentials

# 시크릿을 사용하여 Google Cloud에 인증
secret = get_secret()
credentials = google_auth(secret)

client = speech.SpeechClient(credentials=credentials)

app.secret_key = secrets.token_hex(16)
cluster = MongoClient(os.environ.get("MONGODB_URL"))

db = cluster["voice_ai_app"]


def synthesize_speech(text, language_code):
    name = "ja-JP-Neural2-B"
    if language_code == "en-US":
        name="en-US-Studio-O"
    startTime= time.time()
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, 
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        name = name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input, 
        voice=voice, 
        audio_config=audio_config
    )
    endTime = time.time()
    elapse = endTime - startTime
    
    startTime= time.time()
    audio_base64_bytes = base64.b64encode(response.audio_content)
    audio_base64_string = audio_base64_bytes.decode('ascii')
    endTime = time.time()
    elapse = endTime - startTime
    
    return (audio_base64_string)


def chat(user_input , prevDialog):
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt= prevDialog+f"User: {user_input}\nAi: ""\nja-JP",
            temperature=0.5,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=1.0,
            presence_penalty=1.0,
            n=1,
            stop=["User:"],
        )
        ai_response = response.choices[0].text.strip()
        return ai_response
    except Exception as e:
        return str(e)

@app.route("/contextstart",methods=['POST'])
def contextstart():
    language_code = request.json.get('input')
    response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Let's set random situation and engage in talk with me. talk to me anything.\nUse language {language_code}.",
                temperature=0.5,
                max_tokens=150,
                top_p=1.0,
                frequency_penalty=1.0,
                presence_penalty=1.0,
                n=1,
                stop=["User:"],)
    ai_response = response.choices[0].text.strip()
    audio = synthesize_speech(ai_response,language_code)
    return(jsonify({"ai_response":ai_response,"audio": audio}))

@app.route('/signup', methods=['POST'])
def signup():
    id = request.json.get('id')
    password = request.json.get('password')
    name = request.json.get('name')
    if db.users.find_one({"id": id}):
        return jsonify({"message": "Fail"})
    else : 
        db.users.insert_one({"id": id, "password": password, "name": name, "language": "ja-JP", "contextMode" : 0 ,"list":[], "credit":0, "item":[], "equip":0})
        return jsonify({"message":"Success"})

@app.route('/login', methods=['POST'])
def login():
    id = request.json.get('id')
    password = request.json.get('password')
    user = db.users.find_one({"id": id, "password": password})
    if user:
        return jsonify({"message" : "Success", "id" : id, "language" : user["language"], "contextMode" : user["contextMode"], "list":user["list"], "credit":user['credit'], "item":user['item'], "equip":user['equip']})
    else :
        return jsonify({"message" : "Fail"})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file found"}), 400
    language_code = request.form.get('languageCode')
    startTime = time.time()
    transcription_text = ""
    pronunciation = 0
    prevDialog = request.form.get('prevDialog')
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)
    endTime = time.time()
    elapse = endTime - startTime
    
    startTime = time.time()
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(48000)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    audio.export(file_path, format="wav")
    endTime = time.time()
    elapse = endTime - startTime
    
    startTime = time.time()
    with open(file_path, "rb") as audio_file:
        content = audio_file.read()
        audio = speech.RecognitionAudio(content=content)
    endTime = time.time()
    elapse = endTime - startTime
    
    startTime = time.time()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code=language_code,
    )
    response = client.recognize(config=config, audio=audio)
    endTime = time.time()
    elapse = endTime - startTime
    
    startTime = time.time()
    for result in response.results:
        transcription_text = result.alternatives[0].transcript
        pronunciation = result.alternatives[0].confidence

    chat_response = chat(transcription_text ,prevDialog)
    endTime = time.time()
    elapse = endTime - startTime
    
    audio = synthesize_speech(chat_response, language_code)
    return jsonify({"sttResponse": transcription_text, "chatResponse": chat_response, "audio": audio, "pronunciation" : pronunciation}), 200

@app.route('/grammer', methods=['POST'])
def grammer():
    input = request.json.get('input')
    id = request.json.get('id')
    user = db.users.find_one({"id": id})
    language = user['language']
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Correct grammer in standard {language}:\n\n {input}.",
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    output = response.choices[0].text.strip().split('\n')
    if len(output) <= 2:
        return jsonify({"grammer": output[0]})
    else:
        return jsonify({"grammer": output[2]})

@app.route('/score', methods=['POST'])
def score():
    input = request.json.get('input')
    input2 = request.json.get('input2')
    # 
    letters = ('?', ',', '.')
    replacements = ('', '', '')
    table = input2.maketrans(dict(zip(letters, replacements)))
    input2 = input2.translate(table)
    ratio = SequenceMatcher(None, input,input2 ).ratio()
    return jsonify({"grammer_score" : ratio})

@app.route('/fetch', methods=['POST'])
def language():
    id = request.json.get('id')
    user = db.users.find_one({"id": id})
    return jsonify({"language":user['language'] , "context": user['contextMode'], "list":user['list'], "credit":user['credit'], "item":user['item'], "equip":user['equip']}), 200

@app.route('/change_language',methods =['POST'] )
def change_language():
    id = request.json.get('id')
    language = request.json.get('language')
    db.users.update_one({"id": id}, {"$set": {"language": language}})
    user = db.users.find_one({"id": id})
    return jsonify({"language":user['language']}),200

@app.route('/flow_flag', methods=['POST'])
def get_flow_flag():
    id = request.json.get('id')
    flow_flag = request.json.get('flow_flag')
    db.users.update_one({"id": id}, {"$set": {"contextMode": flow_flag}})
    user = db.users.find_one({"id": id})
    return jsonify({"contextMode":user['contextMode']}),200

@app.route('/update_list',methods =['POST'] )
def update_list():
    id = request.json.get('id')
    list = request.json.get('list')
    db.users.update_one({"id": id}, {"$set": {"list": list}})
    return jsonify({"message":"success"}),200

@app.route('/context', methods=['POST'])
def context():
    aisentence = request.json.get('aisentence')
    usersentence = request.json.get('usersentenceinput')
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"{aisentence}\n{usersentence}\nIf the above two sentences are contextual, please just return 1,  if not enter 0",
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
        
    output = response.choices[0].text.strip().split('\n')
    output = [elem for elem in output if elem.isdigit()]
    output = int(output[0])
    
    return jsonify({"output": output})

@app.route('/update-credits', methods=['POST'])
def update_credits():
    id = request.json.get('id')
    credits = request.json.get('credits')
    db.users.update_one({"id": id}, {"$set": {"credit": credits}})
    return jsonify({"message": "Success"})

@app.route('/update-purchase', methods=['POST'])
def update_purchase():
    id = request.json.get('id')
    items = request.json.get('items')
    db.users.update_one({"id": id}, {"$push": {"item": items}})
    return jsonify({"message": "Success"})

@app.route('/equip', methods=['POST'])
def equip():
    id = request.json.get('id')
    equip = request.json.get('equip')
    db.users.update_one({"id": id}, {"$set": {"equip": equip}})
    return jsonify({"message": "Success"})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
