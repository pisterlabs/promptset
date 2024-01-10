from flask import Blueprint
from flask import render_template, request
from setting import socketio
from transform import pcm2wav
from AI_chat import openai_client
from api_key import DIFY_API_KEY_SPEAKING
from stream_generation import generate_dify_speaking
import base64, json
from assess2 import send_to_assess
from login_check import check_login

import requests

speaking_grader_bp = Blueprint('speaking_grader', __name__)

file3 = {}

@speaking_grader_bp.route('/upload',methods=['GET'])
def upload():
    user_info = check_login()  # check which user is logged in.

    if user_info:  # if there is a logged-in user
        return render_template("upload.html", current_user=user_info.login_name)
    else:
        return render_template("upload.html")

@speaking_grader_bp.route('/file',methods=['POST'])
def save_file():
    data = request.files

    file = data['file']

    form = request.form
    current_user = form.get('user')

    buffer_data = file.read()
    with open("static/en/"+"audio"+current_user+file.filename[-4:], 'wb+') as f:
        f.write(buffer_data)

    form=1
    if file.filename[-4:]==".pcm":
        form=1
    elif file.filename[-4:]==".wav":
        form=2
    elif file.filename[-4:]==".mp3":
        form=3

    result=send_to_assess(base64.b64encode(buffer_data).decode(),form)
    print(result)

    f.close()
    audio_file = open("static/en/"+"audio"+current_user+file.filename[-4:], "rb")

    transcript = openai_client[current_user].audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )
    audio_file.close()
    print(transcript)

    # socketio.emit('reply5', transcript+"\n")
    socketio.emit('reply5', {"user": current_user, "result": transcript + "\n"})

    # socketio.emit('reply5', 'PronAccuracy: ' + str(json.loads(result)['PronAccuracy']) + "\nPronFluency: " + str(
    #     json.loads(result)['PronFluency'])+"\n\nWaiting for the response...\n\n")
    socketio.emit('reply5', {"user": current_user, "result": 'PronAccuracy: ' + str(
        json.loads(result)['PronAccuracy']) + "\nPronFluency: " + str(
        json.loads(result)['PronFluency']) + "\n\nWaiting for the response...\n\n"})

    headers = {'Authorization': DIFY_API_KEY_SPEAKING, 'Content-Type': 'application/json'}
    payload = {'inputs': {}, 'query': transcript, 'response_mode': 'streaming',
               'conversation_id': '', 'user': current_user}

    response = requests.post("https://api.dify.ai/v1/chat-messages", data=json.dumps(payload),
                             headers=headers)
    generate_dify_speaking(response,current_user)

    return ""

@socketio.on('play3')
def play3(req):
    print(req['current_user'])
    if req['current_user'] not in file3:
        file3[req['current_user']]=open("static/en/e3xample"+req['current_user']+".pcm", "wb")

@socketio.on('update3')
def update3(req):

    if req['current_user'] not in file3:
        return

    if 'stop' in req:

        file_popped = file3.pop(req['current_user'])
        file_popped.close()

        file_path = r"static/en/e3xample"+req['current_user']+".pcm"
        to_path = r"static/en/e3xample"+req['current_user']+".wav"
        pcm2wav(file_path, to_path)

        audio_file = open("static/en/e3xample"+req['current_user']+".wav", "rb")

        transcript = openai_client[req['current_user']].audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )

        if 'type' in req:

            socketio.emit('reply5', {"user": req['current_user'], "result": transcript + "\n"})

            with open("static/en/e3xample"+req['current_user']+".wav", "rb") as f:
                base64_data = base64.b64encode(f.read()).decode()
                print(type(base64_data))
                result = send_to_assess(base64_data,2)
                socketio.emit('reply5', {"user": req['current_user'],"result": 'PronAccuracy: ' + str(json.loads(result)['PronAccuracy']) + "\nPronFluency: " + str(
        json.loads(result)['PronFluency'])+"\n\nWaiting for the response...\n\n"})

            headers = {'Authorization': DIFY_API_KEY_SPEAKING, 'Content-Type': 'application/json'}
            payload = {'inputs': {}, 'query': transcript, 'response_mode': 'streaming',
                       'conversation_id': '', 'user': req['current_user']}

            response = requests.post("https://api.dify.ai/v1/chat-messages", data=json.dumps(payload),
                                     headers=headers)

            generate_dify_speaking(response,req['current_user'])

        else:
            socketio.emit('reply3', {"user": req['current_user'], "result": " "+transcript, "end": 1})
    else:
        file3[req['current_user']].write(req['data'])