import openai
from ChatGPT import BOT_KEY
from translate import Translator
import requests
import urllib.parse
from voicevox import Client 
import asyncio

openai.api_key = BOT_KEY

messages = []

file = open('initiator.txt', 'r')
prompt = file.read()

def translate_text(text, target_language):
        translator = Translator(to_lang=target_language)
        translation = translator.translate(text)
        return translation
    
client = Client("localhost", 50021)

base_url = "http://localhost:50021"

def generate_speech(text):
      # generate initial query
      speaker_id = '14'
      params_encoded = urllib.parse.urlencode({'text': text, 'speaker': speaker_id})
      r = requests.post(f'{base_url}/audio_query?{params_encoded}')
      voicevox_query = r.json
      voicevox_query['volumeScale'] = 4.0
      voicevox_query['intionationScale'] = 1.5 
      voicevox_query['prePhonemeLength'] = 1.0
      voicevox_query['postPhonemeLength'] = 1.0

      # synthesize voice as wav file
      params_encoded = urllib.parse.urlencode({'speaker': speaker_id})
      r = requests.post(f'{base_url}/synthesis?{params_encoded}', json=voicevox_query)  

      return r.content

messages.append({"role": "system", "content": prompt})
print("Type to start\n")
while input != "quit()":
    message = input()
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = messages
    )

    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    
    text = reply
    target_language = "ja"
    translated_text = translate_text(text,target_language)

    audio_data = generate_speech(translated_text)
    print("\n" + reply + "\n")

