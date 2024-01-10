from decouple import config
import requests
import json
import time
import openai
from functions.db import get_recent_messages

base_url = "https://api.assemblyai.com/v2"

headers = {
    "authorization": config("ASSEMBLY_AI_API_KEY")
}

openai.organization = config("OPEN_AI_ORG")
openai.api_key = config("OPEN_AI_API_KEY")


def upload_audio_to_assembly(audio_file):
    try:
        response = requests.post(base_url + "/upload", headers=headers, data=audio_file)

        upload_url = response.json()["upload_url"]
        data = {
        "audio_url": upload_url # You can also use a URL to an audio or video file on the web
        }
        transcription_id = get_transcription_id(data)
        return transcription_id
    except Exception as e:
        print(e)
        return
    

def get_transcription_id(data):
    try:
        url = base_url + "/transcript"
        response = requests.post(url, json=data, headers=headers)
        return response.json()['id']
    except Exception as e:
        print(e)
        return



def convert_audio_to_text(audio_file):
    try:
        transcript_id = upload_audio_to_assembly(audio_file)
        polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

        while True:
            transcription_result = requests.get(polling_endpoint, headers=headers).json()

            if transcription_result['status'] == 'completed':
                break

            elif transcription_result['status'] == 'error':
                raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

            else:
                time.sleep(3)
        return transcription_result
    except Exception as e:
        print(e)
        return
    

def get_chat_response(message_input):

  messages = get_recent_messages()
  user_message = {"role": "user", "content": message_input }
  messages.append(user_message)

  try:
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages
    )
    message_text = response["choices"][0]["message"]["content"]
    return message_text
  except Exception as e:
    print(e)
    return