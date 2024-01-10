import requests
import json
import random 
import openai
from halo import Halo
import config

ELEVENLABS_API_KEY = config.ELEVENLABS_API_KEY
openai.api_key = config.OPENAI_API_KEY


headers = {
    'accept': 'audio/mpeg',
    'xi-api-key': ELEVENLABS_API_KEY,
    'Content-Type': 'application/json'
}

headersVoices = {
    'accept': 'application/json',
    'xi-api-key': ELEVENLABS_API_KEY
}

responseVoices = requests.get('https://api.elevenlabs.io/v1/voices', headers=headersVoices)

voiceContent = (responseVoices.content).decode("utf-8")
dataDict = json.loads(voiceContent)

nameIDDict = {}

for x in dataDict['voices']:
    name = x ['name']
    voiceID = x ['voice_id']
    nameIDDict[name] = voiceID
    
spinner = Halo(text='Loading', spinner='dots')

def generateSpeech(voiceID, theme):
    if voiceID == 10: 
        choice = random.choice(list(nameIDDict.values()))
    else:
        choice = (list(nameIDDict.values()))[voiceID]
    
    initPrompt = """ You are a speech assistant, you are tasked with creating a speech for a client. 
    Write clearly, consicely and in simple language"""
    
    speechPrompt = f"""Write a speech in the theme of {theme}"""
    
    spinner.start("Text Generating")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "system", "content": initPrompt},
        {"role": "user", "content": speechPrompt}]
    )
    spinner.succeed("Generation Complete")
    
    response = completion['choices'][0]["message"]["content"]



    json_data = {
        'text': response,
        'voice_settings': {
            'stability': 0,
            'similarity_boost': 0,
        },
    }

    spinner.start("Converting to Audio")    
    response = requests.post(f'https://api.elevenlabs.io/v1/text-to-speech/{choice}', headers=headers, json=json_data)
        
    content = response.content    
    with open("files/output.mp3", "wb") as file:
            file.write(content)
    
    spinner.succeed("Complete")