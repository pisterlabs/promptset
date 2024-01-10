from flask import Blueprint, request, jsonify, redirect, url_for
import requests
import json
import os
import hashlib
import base64
import subprocess
import openai
from gtts import gTTS

from config import base_url, api_token, org_id, openai_apikey

# Create a Blueprint object
file_bp = Blueprint('file', __name__)

openai.api_key = openai_apikey

UPLOAD_FOLDER = "/backend/code/assets"
SRT_FOLDER = "/backend/code/subtitled"

headers = {
    "x-bv-org-id": org_id,
    "Content-Type": "application/json",
    "Accept": "application/json",
    "authorization": "Bearer " + api_token,
}

# list videos under library/video
@file_bp.route('/list', methods=['GET'])
def list():
    url = base_url + "/bv/cms/v1/library/files"
    querystring = {"current_page":"1","items_per_page":"10","type":"FILE_TYPE_VIDEO"}
    
    if len(request.args) != 0 and any(key not in ['video_name'] for key in request.args):
        return jsonify({
            "code":"2",
            "message":"invalid arguments",
        }), 400
    elif len(request.args) == 0:
        # if no specify video name, then return all the files
        response = requests.get(url, headers=headers, params=querystring)
        return response.json(), 200
    
    querystring["filter.name"] = request.args["video_name"] # query the requested video_name
    response = requests.get(url, headers=headers, params=querystring)
    return response.json(), 200

# search for specific file
@file_bp.route('/search/<video_name>', methods=['GET'])
def search(video_name):
    if video_name[-4:] != ".mp4":
        return jsonify({
            "code":"2",
            "message":"invalid arguments",
        }), 400
    list_url = base_url + "/bv/cms/v1/library/files"
    querystring = {"current_page":"1","items_per_page":"5","type":"FILE_TYPE_VIDEO","filter.name":video_name}
    
    response = requests.get(list_url, headers=headers, params=querystring)
    #return response.json()
    if response.json()['pagination']["total_items"] ==0:
        return jsonify({
            "code":"4",
            "message":"search file not found",
        }), 400 

    file = response.json()["files"][0]
    
    search_url = base_url + "/bv/cms/v1/library/files/" + file["id"]

    response = requests.get(search_url, headers=headers)

    return response.json(), 200

def get_file_id(video_name):
    url = base_url + "/bv/cms/v1/library/files"
    querystring = {"current_page":"1","items_per_page":"1","type":"FILE_TYPE_VIDEO","filter.name":video_name}

    response = requests.get(url, headers=headers, params=querystring)
    return response.json()['files'][0]['id']

def get_srt_id(srt_name):
    url = base_url + "/bv/cms/v1/library/files"
    querystring = {"current_page":"1","items_per_page":"1","type":"FILE_TYPE_SUBTITLE","filter.name":srt_name}

    response = requests.get(url, headers=headers, params=querystring)
    return response.json()['files'][0]['id']

def sha1_digest(file_path):
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as file:
        while True:
            data = file.read(65536)  # Read the file in 64KB chunks
            if not data:
                break
            sha1.update(data)

    sha1_hash = sha1.digest()
    base64_encoded_hash = base64.b64encode(sha1_hash).decode('utf-8')
    return base64_encoded_hash


# This function can generate subtitle srt file of input file
def generate_subtitle(video_path):
    url = "http://whisper-api:9000/asr"
    params = {"task":"transcribe", "output":"srt"}

    video_file = open(video_path, 'rb')
    body = {"audio_file":video_file}
    response = requests.post(url, params=params, files=body)
    video_file.close()

    if response.status_code != 200:
        return "Failed to transcribe video"
    
    srt_path = "/backend/code/subtitled/" + os.path.basename(video_path).rsplit(".")[0] + ".srt"
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        srt_file.write(response.content.decode('utf-8'))

    subtitle_video = "/backend/code/assets/" + os.path.basename(video_path).rsplit(".")[0] + "_sub.mp4"

    ffmpeg_command = f"ffmpeg -i {video_path} -vf subtitles='{srt_path}' -c:a copy {subtitle_video}"

    try:
        # Run the FFmpeg command
        subprocess.run(ffmpeg_command, check=True, shell=True)
        return subtitle_video
    except subprocess.CalledProcessError as e:
        print(f"Error adding subtitles: {e}")
    return subtitle_video

# get respone from chatgpt and create chatting video


def gpt_talk(video_path):
    url = "http://whisper-api:9000/asr"
    
    params = {"task":"transcribe", "output":"txt"}
    video_file = open(video_path, 'rb')
    body = {"audio_file":video_file}
    response = requests.post(url, params=params, files=body)
    video_file.close()
    if response.status_code != 200:
        return "Failed to transcribe video"
    
    srt_path = "/backend/code/subtitled/" + os.path.basename(video_path).rsplit(".")[0] + ".txt"
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        srt_file.write(response.content.decode('utf-8'))
    
   
    with open(srt_path, 'r', encoding='utf-8') as file:
        lines = file.read().splitlines()
        text = '\n'.join(line for line in lines if not line.isdigit())
    
    # Make a request to OpenAI for a response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        temperature=0.9,
        max_tokens=150,  # Adjust the number of tokens as needed
    )
    
    text = response.choices[0].text
    #language = 'en'
    filepath = "/backend/code/audio/" + os.path.basename(srt_path).rsplit(".")[0] + ".mp3"
    tts = gTTS(text=text)
    tts.save(filepath)

    return filepath


# This function implements the part to upload files to BV library
def file_upload(file_path):
    upload_url = base_url + "/bv/cms/v1/library/files:upload"

    file_type = "FILE_TYPE_VIDEO"
    file_extension = os.path.splitext(file_path)[1].lstrip(".")
    if file_extension == "srt":
        file_type = "FILE_TYPE_SUBTITLE"

    payload = { "file": {
        "name": os.path.basename(file_path),
        "size": str(os.stat(file_path).st_size),
        "source": "FILE_SOURCE_UPLOAD_IN_LIBRARY",
        "type": file_type
    } }

    upload_response = requests.post(upload_url, json=payload, headers=headers)
    return upload_response

# This function implements the complete signal to send when finish file uploading
def upload_complete(upload_res, srt):
    fid = upload_res.json()['file']['id']
    upload_id = upload_res.json()['upload_data']['id']

    folder = UPLOAD_FOLDER
    if srt == True:
        folder = SRT_FOLDER

    # generate checksum_sha1
    checksum_sha1 = sha1_digest(folder + "/" + upload_res.json()['file']['name'])

    # generate body for complete upload
    parts = upload_res.json()['upload_data']['parts']
    all_res = []
    i = 1
    for part in parts:
        with open(folder + "/" + upload_res.json()['file']['name'], 'rb') as f:
            part_res = requests.put(part['presigned_url'], data=f)
            all_res.append({
                "etag": part_res.headers['ETag'],
                "part_number": i
            })
        i += 1
    
    # complete file upload url
    complete_url = base_url + "/bv/cms/v1/library/files/" + fid + ":complete-upload"

    payload = { "complete_data": {
        "checksum_sha1": checksum_sha1,
        "id": upload_id,
        "parts": all_res,
    } }

    complete_response = requests.post(complete_url, json=payload, headers=headers)
    return complete_response

# upload file
@file_bp.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({
            "code":"2",
            "message":"No file part",
        }), 400
    
    # TODO : require filename checking
    # Write your code here

    file = request.files['file']
    if file.filename == '' or file.filename[-4:]!=".mp4":
        return jsonify({
            "code":"3",
            "message":"No selected file",
        }), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    filename = file.filename
    file.save(file_path) # store file in local

    result = ""
    if request.values.get('subtitle') == "1":
        file_path = generate_subtitle(file_path)
        filename = os.path.basename(file_path)
        audiopath = gpt_talk(os.path.join(UPLOAD_FOLDER, file.filename))
        # audiopath = gpt_talk("/backend/code/subtitled/" + file.filename.rsplit(".")[0] + ".srt")

    upload_res = file_upload(file_path)
    if upload_res.status_code != 200:
        return jsonify({
            "code":"2",
            "message":"Error when uploading file",
            "path":file_path,
            "message_BV":upload_res.json()
        }), 400

    complete_res = upload_complete(upload_res, False)
    if complete_res.status_code != 200:
        return jsonify({
            "code":"3",
            "message":"Error when completing upload"
        }), 400

    # create VOD for the file
    return redirect(url_for('vod.create', video_name = filename))

    # return redirect(url_for('index'))