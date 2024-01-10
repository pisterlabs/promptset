import requests
import os
from tempfile import NamedTemporaryFile
from ..environment import openai,client

# Environment variables should be used to securely store the API keys
def get_audio(LINE_ACCESS_TOKEN,OPENAI_APIKEY,message_id):
    url = f'https://api-data.line.me/v2/bot/message/{message_id}/content'

    headers = {
        'Authorization': f'Bearer {LINE_ACCESS_TOKEN}',
    }

    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code == 200:
        # Save the audio file temporarily
        with NamedTemporaryFile(suffix=".m4a", delete=False) as temp:
            temp.write(response.content)
            temp.flush()

        # Call the speech_to_text function with the temporary file
        return speech_to_text(temp.name)
    else:
        print(f"Failed to fetch audio: {response.content}")
        return None

def speech_to_text(file_path):
    with open(file_path, 'rb') as f:
        response = client.audio.transcriptions.create(
            model="whisper-1", 
            file=f
            )

        if response.status_code == 200:
            return response.json().get('text')
        else:
            print(f"Failed to transcribe audio: {response.content}")
            return None