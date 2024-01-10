from openai import OpenAI
import sys
import json
from elevenlabs import *

### This is a test file!
### This script stream the token from ChatGPT's completion to ElevenLabs's API.
### This results in faster or maybe some times real-time processing by ElevenLabs, however, it may compromise the quality of the output.
### This script does not have a turn-by-turn conversation

try:
    envFile = open('.env', "r").read()
except:
    print('Error while trying to open .env file.')
    exit(False)

try:
    env = json.loads(envFile)
except:
    print("Can't decode JSON data in .env file")
    exit(False)

client = OpenAI(
    api_key=env["openai"]
    )

set_api_key(env["elevenlabs"])

msg = input("Me : ")

completion = client.chat.completions.create(
    model = "gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": msg}
    ],
    temperature=0.8,
    stream=True
)

print("ChatGPT : ", end='')

def streamCompletion(completion):
    for chunk in completion:
        if chunk.choices[0].finish_reason == "stop":
            yield(' ')
            continue

        print(chunk.choices[0].delta.content, end="")
        sys.stdout.flush()
        yield chunk.choices[0].delta.content

audio = generate(text=streamCompletion(completion), voice=Voice(voice_id=env['voiceID']), model="eleven_multilingual_v2", stream=True)
stream(audio)


### TO-DO
'''
while True:

    msg = input("Moi : ")

    completion = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": msg}
        ],
        temperature=0.8,
        stream=True
    )

    print("ChatGPT : ", end='')

    for chunk in completion:
        if chunk.choices[0].finish_reason == "stop":
            print("")
            continue
        
        print(chunk.choices[0].delta.content, end='')
        sys.stdout.flush()

'''