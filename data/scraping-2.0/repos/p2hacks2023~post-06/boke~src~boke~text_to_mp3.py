import json
import time
import tqdm
from pathlib import Path
from openai import OpenAI
client = OpenAI()

json_open = open("data/dajare.json")
json_load = json.load(json_open)
dajare_list = json_load["contents"]

for dajare in tqdm.tqdm(dajare_list):

    speech_file_path = "data/speech/" + dajare["dajare"] + ".mp3"
    response = client.audio.speech.create(
    model="tts-1-hd",
    voice="alloy",
    input=dajare["dajare"]
    )

    response.stream_to_file(speech_file_path)

    time.sleep(20) # apiの制限に引っかからないようにするために20秒待つ 