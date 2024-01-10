import requests
import time
import json
from decouple import config
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import os
import openai
import datetime


openai.organization = config("OPEN_AI_ORG")
openai.api_key = config("OPEN_AI_KEY")
API = config("API")
ELEVEN_LABS_API_KEY = config("ELEVEN_LABS_API_KEY")


def video_avatar(speech):
    url = 'https://api.heygen.com/v1/video.generate'
    api_key = API

    headers = {
        'X-Api-Key': api_key,
        'Content-Type': 'application/json'
    }

    data = {
        "background": "#ffffff",
        "clips": [
            {
                "avatar_id": "Daisy-inskirt-20220818",
                "avatar_style": "normal",
                "input_audio": speech,
                "offset": {
                    "x": 0,
                    "y": 0
                },
                "scale": 1,
                "voice_id": "1bd001e7e50f421d891986aad5158bc8",
                "voice_language": "Spanish"
            }
        ],
        "ratio": "16:9",
        "test": True,
        "version": "v1alpha"
    }

    print("tope 2")
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        video_id = result['data']['video_id']
        print(video_id)
        result = response.content
        print(result)
        print("///secuencia 1 hecha")
    else:
        print('Error en la solicitud:', response.status_code)
    return video_id


def video_avatar_texto(speech):
    url = 'https://api.heygen.com/v1/video.generate'
    api_key = API

    headers = {
        'X-Api-Key': api_key,
        'Content-Type': 'application/json'
    }

    data = {
        "background": "#ffffff",
        "clips": [
            {
                "avatar_id": "Daisy-inskirt-20220818",
                "avatar_style": "normal",
                "input_text": speech,
                "offset": {
                    "x": 0,
                    "y": 0
                },
                "scale": 1,
                "voice_id": "1bd001e7e50f421d891986aad5158bc8",
            }
        ],
        "ratio": "16:9",
        "test": True,
        "version": "v1alpha"
    }


    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        print(result)
        video_id = result['data']['video_id']
        print(video_id)
        result = response.content
        print(result)
        print("///secuencia 1 hecha")
    else:
        print(response.content)
        print('Error en la solicitud:', response.status_code)
    return video_id



def url_video(video_id):
    url = f'https://api.heygen.com/v1/video_status.get?video_id={video_id}'
    api_key = API

    headers = {
        'X-Api-Key': api_key
    }

    while True:
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            result = response.json()
            print(result)
            message = result['data']['status']
            if message == 'processing':
                time.sleep(5)
            elif message == 'completed':
                video_url = result['data']['video_url']
                print(video_url)
                print("////secuencia 2 hecha")
                # Manejar la respuesta de la API según tus necesidades
                print(result)
                return video_url
            else:
                print('Error en la solicitud:', message)
                break
        else:
            print('Error en la solicitud:', response.status_code)
            break

def download_video(url):
    download_url = url  # Reemplaza con la URL de descarga del archivo
    output_file = 'first_video.mp4'  # Nombre del archivo de salida

    response = requests.get(download_url)

    if response.status_code == 200:
        with open(output_file, 'wb') as file:
            file.write(response.content)
        print(f'Archivo descargado exitosamente: {output_file}')
        return response.content
    else:
        print('Error en la descarga:', response.status_code)
        return response.status_code

def get_audio_download_url(file_path):
    storage_client = storage.bucket("samai-b9f36.appspot.com")
    blob = storage_client.blob(file_path)
    return blob.generate_signed_url(method="GET", expiration=datetime.timedelta(days=1))


def convert_text_to_speech_video(texto,user):

    CHUNK_SIZE = 1024
    #Define Data
    body = {
        "text": texto,
        "model_id": "eleven_multilingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost":0,
        }
    }

    #Define voice
    voice_rachel = "21m00Tcm4TlvDq8ikWAM"
    voice_antoni = "oUciFfPUJCaDqHitPLu5"


    #Constructing Headers and Endpoint
    headers = {"xi-api-key": ELEVEN_LABS_API_KEY, "Content-Type": "application/json","accept": "audio/mpeg"}
    endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_antoni}"

    # Send request
    try:
        response = requests.post(endpoint, json=body, headers=headers)
    except Exception as e:
        return
    
    #Handle Response 
    if response.status_code == 200:
        with open('output.mp3', 'wb') as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

        storage_client = storage.bucket("samai-b9f36.appspot.com")
        folder_name = f"{user}/{user}_audio"
        os.makedirs(folder_name, exist_ok=True)
        file_name = f"{folder_name}/output.mp3"

        blob = storage_client.blob(file_name)
        blob.upload_from_filename('output.mp3')

        # Obtener la URL del archivo de audio en Firebase Storage
        audio_url = get_audio_download_url(file_name)
        print(audio_url)

        print("exito")
        return audio_url
    else:
        return
    

def get_chat_response_video(message_input):

    messages = get_recent_messages_video()
    user_messages = {"role":"user","content":message_input}
    messages.append(user_messages)
    print(messages)

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
    
def get_recent_messages_video():

  #asignar variable
  prompt_usuario = "Tu nombre es Samantha. Tus respuestas deben ser muy cortas, menos de 10 palabras"
  
  learn_instruction = {"role": "system", 
                       "content": prompt_usuario + " Keep your answers under 10 words"}
  
  # Initialize messages
  messages = []

  # Add Random Element
  # x = random.uniform(0, 1)
  # if x < 0.2:
  #   learn_instruction["content"] = learn_instruction["content"] + "Your response will have some sarcastic humour. "
  # elif x < 0.5:
  #   learn_instruction["content"] = learn_instruction["content"] + "Your response will be in a rude "
  # else:
  #   learn_instruction["content"] = learn_instruction["content"] + "Your response will have some dark humour "

  # Append instruction to message
  messages.append(learn_instruction)

  
  # Return messages
  return messages