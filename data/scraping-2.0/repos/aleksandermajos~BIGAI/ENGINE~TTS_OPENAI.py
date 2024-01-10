from openai import OpenAI
from playsound import playsound
import os
from pathlib import Path
p = Path.cwd()
path_beginning = str(p.home())+'/PycharmProjects/BIGAI/'
path = path_beginning+"DATA/ALOHA/"
cwd = os.getcwd()
f = open(path+"account.txt", "r")
client = OpenAI(api_key=f.read())

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="onyx",
    input="Pan kleks ruchal panny mlode"

)

response.stream_to_file("output.mp3")
playsound('output.mp3')
