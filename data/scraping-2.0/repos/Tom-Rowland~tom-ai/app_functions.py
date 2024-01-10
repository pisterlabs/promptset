import requests, time, hashlib
from config import OPENAI_TOKEN, RESEMBLEAI_TOKEN, RESEMBLEAI_PROJECTID,RESEMBLEAI_VOICEID
import database

def generate_response(chat_history):
    url = 'https://api.openai.com/v1/completions'

    headers = {
        'Content-Type':'application/json',
        'Authorization':'Bearer '+ OPENAI_TOKEN}
    data = {
        'model':'text-davinci-003',
        'prompt': chat_history,
        'max_tokens': 30,
        'temperature': 0.5,
        'top_p':1,
        'n':1}
    response = requests.post(url, json=data, headers=headers)
    chat_response = response.json()['choices'][0]['text']
    return chat_response

def create_clip(body):
    url = f"https://app.resemble.ai/api/v2/projects/{RESEMBLEAI_PROJECTID}/clips"

    headers = {
        'Content-Type':'application/json',
        'Authorization':'Token token='+ RESEMBLEAI_TOKEN}

    data = {
        'title': body[:256],
        'body': body,
        'voice_uuid': RESEMBLEAI_VOICEID,
        'is_public': False,
        'is_archived': False,
        'callback_uri': 'https://'}

    response = requests.post(url, json=data, headers=headers)
    clip_id = response.json()['item']['uuid']

    return clip_id, body

def get_clip(clip_id, body):
    url = f"https://app.resemble.ai/api/v2/projects/{RESEMBLEAI_PROJECTID}/clips/{clip_id}"
    print(url)
    headers = {
        'Content-Type':'application/json',
        'Authorization':'Token token='+ RESEMBLEAI_TOKEN}

    for i in range(30):
        response = requests.get(url, headers=headers)
        print(i)
        if 'audio_src' in response.json()['item'].keys():
            audio_file_key = database.write_audio_file(body, clip_id)
            return response.json()['item']['audio_src'], audio_file_key
        time.sleep(1)
    clip_id, body = create_clip(body)
    get_clip(clip_id, body)

def check_clip_already_exists(body):
    body_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
    url = database.db_query(f"SELECT TOP 1 [ClipId] FROM [tom-ai].[dimAudioFile] WHERE [PhraseHash] = '{body_hash}'").values.tolist()
    if url:
        return url[0][0]
    else:
        return None