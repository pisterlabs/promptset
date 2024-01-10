import requests
from pathlib import Path
import openai 
from decouple import config



ELEVEN_LABS_API_KEY = config("ELEVEN_LABS_API_KEY")
openai.organization = config('OPEN_AI_ORG')
openai.api_key = config('OPEN_AI_KEY')


#https://elevenlabs.io/docs/api-reference/get-voices
def convert_text_to_speech(message):

    #Define Data (body)
    body = {
        "text": message,
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }

    #Define voice
    rachel_voice = "21m00Tcm4TlvDq8ikWAM"

    #Define Data (header)
    headers ={"xi-api-key": ELEVEN_LABS_API_KEY, "Content-Type": "application/json", "accept": "audio/mpeg"}
    endpoint = "https://elevenlabs.io/docs/api-reference/text-to-speech/v1/text-to-speech/{rachel_voice}"

    #Send Request
    try:
        response = requests.request("POST", endpoint, json=body,  headers=headers)

    except Exception as e:
        return
    
    #handle response 
    if response.status_code == 200:
        return response.content
    else:
        return