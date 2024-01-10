from dotenv import load_dotenv
import openai
import requests
import os

load_dotenv()

azure_key = os.getenv('AZURE_KEY')


url = "https://westus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=detailed"
headers = {
    "Accept": "application/json;text/xml",
    "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000",
    "Ocp-Apim-Subscription-Key": azure_key,
}

with open('test1.wav', 'rb') as audio_file:
    response = requests.post(url, headers=headers, data=audio_file)

if response.status_code == 200:
    print("Request successful!")
    print(response.json())
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)
