import openai
import os
from os.path import join, dirname
from dotenv import load_dotenv
import tempfile
import sys


dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = API_KEY


def generate_response(user_input, messages):
    text = sst(user_input)
    question = {'role': 'user', 'content': text}
    messages.append(question)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    try:
        answer = response['choices'][0]['message']['content'].replace('\n', '<br>')
    except:
        answer = 'Error in response'

    return text, answer

def sst(audio):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
        audio.save(temp_audio)
        temp_audio_path = temp_audio.name

    with open(temp_audio_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe(
            file=audio_file,
            model="whisper-1",
            response_format="text",
            language="ja"
        )
    print(transcript, sys.stderr)
    return transcript