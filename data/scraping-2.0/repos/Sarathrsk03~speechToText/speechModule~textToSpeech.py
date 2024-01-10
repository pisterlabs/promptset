from dotenv import load_dotenv
from openai import OpenAI
import os
from os.path import join, dirname


dotenv_path = ".env"
load_dotenv(dotenv_path)

apikey = os.environ.get("openAI_API")

client = OpenAI(api_key=apikey)

def convertTextToSpeech(textToConvert):
    response = client.audio.speech.create(
        model = "tts-1",
        voice = "nova",
        input = textToConvert,
    )

    response.stream_to_file("./static/audio/output.mp3")
    #playsound("output.mp3")

