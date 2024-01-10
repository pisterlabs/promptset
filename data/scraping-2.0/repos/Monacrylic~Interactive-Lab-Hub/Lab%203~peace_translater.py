import argparse
import queue
import sys
import sounddevice as sd
import json
import subprocess
from vosk import Model, KaldiRecognizer

import openai
openai.api_key = "YOUR API KEY (HIDDEN FOR COMMIT)"


q = queue.Queue()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text
    
def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        pass
        # print(status, file=sys.stderr)
    q.put(bytes(indata))

def process_sentence(sentence):
    return f'[Post Processed] {sentence}'


def translate_text(text):
    #prompt = (f"Rewrite the following text to be very humble, polite, and politically correct:\n"
              #f"text: {text}\n")
    prompt = (f"You are a peace translater, I will provide you with a text containing angry ang negative sentiments. And you will translate that text to a positive, friendly, constructive, and polite tone without any swear words:\n"
              f"text: {text}\n"
              f"Your translation of this text: \n")

    response = openai.Completion.create(
        engine="text-davinci-002",
        #engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # print(response.choices)
    peaceful_text = response.choices[0].text.strip()
    return peaceful_text

def speak_sentence(sentence):
    subprocess.check_output(f'./speech-scripts/googletts_arg.sh "{sentence}"', shell=True, stderr=subprocess.PIPE, universal_newlines=True)
    sentence = ''
    
device_info = sd.query_devices(None, "input")
# soundfile expects an int, sounddevice provides a float:
samplerate = int(device_info["default_samplerate"])
    
model = Model(lang="en-us")

with sd.RawInputStream(samplerate=samplerate, blocksize = 8000, device=1,
        dtype="int16", channels=1, callback=callback):
    print("#" * 80)
    print("Press Ctrl+C to stop the recording")
    print("#" * 80)

    rec = KaldiRecognizer(model, samplerate)
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            recognition = json.loads(rec.Result())["text"]
            print(recognition)
            # print(rec.Result())
            processed_sentence = translate_text(recognition)
            print(processed_sentence)
            speak_sentence(processed_sentence)
        else:
            #print(rec.PartialResult())
            pass
