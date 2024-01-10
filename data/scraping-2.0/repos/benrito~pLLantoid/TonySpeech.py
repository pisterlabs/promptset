#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

import speech_recognition as sr

import pyaudio
import wave
import audioop
from collections import deque

from playsound import playsound

from dotenv import load_dotenv

import openai
import requests

import random
import os



FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "recordedFile.wav"
device_index = 6

#define the silence threshold
THRESHOLD = 350
SILENCE_LIMIT = 2 # 2 seconds of silence will stop the recording


AUDIO_FILE = "temp_reco.wav"




USER = "Human"
AGENT = "Plantoid"


# Load environment variables from .env file
load_dotenv()
openai.api_key = os.environ.get("OPENAI")
eleven_labs_api_key = os.environ.get("ELEVEN")


headers = {
    "Accept": "audio/mpeg",
    "Content-Type": "application/json",
    "xi-api-key": eleven_labs_api_key
}

url = "https://api.elevenlabs.io/v1/text-to-speech/o7lPjDgzlF8ZloHzVPeK"

opening_lines = [
        "So tell me, what brings you here?",
        "Would you like to have a little chat with me?",
        "I'm a litte bit busy right now, but happy to entertain you for a bit",
        "I'm eager to get to know you! Tell me something about you.."
        ];

closing_lines = [
        "That's enough, I must return to the blockchain world now. I'm getting low on energy..",
        "You are quite an interesting human, unfortunately, I must go now, I cannot tell you all of my secrets..",
        "I would love to continue this conversation, but my presence is required by other blockchain-based lifeforms..",
        "I'm sorry, I have to go now. I have some transactions to deal with.."
        ];

acknowledgements = [
	"./media/hmm1.mp3",
	"./media/hmm2.mp3",
	];

def default_prompt_config():
    return {
        "model": "gpt-4",
        "temperature": 0.5,
        "max_tokens": 128,
        "logit_bias": {
            198: -100  # prevent newline
        }
    }



def gptmagic(prompt):

    # Prepare the GPT magic

    configs = default_prompt_config()

    # Generate the response from the GPT model
    response = openai.ChatCompletion.create(messages=[{"role": "user", "content": prompt}], **configs)

    messages = response.choices[0].message.content
    print(messages); return messages;




def speaktext(text):
    # Request TTS from remote API
    response = requests.post(url, json={"text": text, "voice_settings": {"stability": 0, "similarity_boost": 0}}, headers=headers)

    if response.status_code == 200:
        # Save remote TTS output to a local audio file with an epoch timestamp
        filename = f"./tonyspeak.mp3"
        with open(filename, "wb") as f:
            f.write(response.content)
        
	#with open(filename, "r") as f:
        # Play the audio file and cleanse
            playsound(filename)
        print("returning from speaktext..." + filename)
        return filename	


def listenSpeech():

    audio = pyaudio.PyAudio()

    print("Im still alive")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
		# input_device_index = device_index,
                frames_per_buffer=CHUNK)


    samples = []

    chunks_per_second = RATE / CHUNK

    silence_buffer = deque(maxlen=int(SILENCE_LIMIT * chunks_per_second))
    samples_buffer = deque(maxlen=int(SILENCE_LIMIT * RATE))

    started = False


### this is for a fixed amount of recording seconds
#    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#        data = stream.read(CHUNK)
#        samples.append(data)




### this is for continuous recording, until silence is reached


    run = 1

    while(run):
        data = stream.read(CHUNK)
        silence_buffer.append(abs(audioop.avg(data, 2)))

        samples_buffer.extend(data)

        if (True in [x > THRESHOLD for x in silence_buffer]):
            if(not started):
                print ("recording started")
                started = True
 #               samples.extend(data)
                samples_buffer.clear()

 #           else:
 #               samples.extend(data)

 #           for x in data:
 #            print(data)
            samples.append(data)

        elif(started == True):
            print ("recording stopped")
            stream.stop_stream()
            
            #hmm = random.choice(acknowledgements)
            #playsound(hmm);

            recwavfile(samples, audio)

            #reset all vars
            started = False
            silence_buffer.clear()
            samples = []

            run = 0


    stream.close()
    audio.terminate()

    return AUDIO_FILE;


def recwavfile(data, audio):

#    print(data)
    wf = wave.open(AUDIO_FILE, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data))
    wf.close()



def recoSpeech(filename):
    with sr.AudioFile(filename) as source:

        r = sr.Recognizer()
        r.energy_threshold = 50
        r.dynamic_energy_threshold = False

        audio = r.record(source)
        usertext = "";
        
        try:
            usertext = r.recognize_google(audio)

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")

        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))


        return usertext



# global variables for the prompting

global prompt_text
turns = []
maxturns = 10 


def updateprompt(turnsss):
    global prompt_text
    lines = []
    transcript = ""
    for turn in turnsss:
        text = turn["text"]
        speaker = turn["speaker"]
        lines.append(speaker + ": " + text)
    transcript = "\n".join(lines)
    return prompt_text.replace("{{transcript}}", transcript)




global opening
global closing

def setup():

    global prompt_text
    opening = random.choice(opening_lines)

    # load the personality of Plantony
    prompt_text = open("/Users/ya/LLMs/PLLantoid/v5/plantony.txt").read().strip()

    print("LOADED ALL prompt :: " + prompt_text)
    return opening

def setdown():

    closing = random.choice(closing_lines); return closing;


if __name__ == "__main__":


	#playsound("/LLMs/PLLantoid/v5/samples/intro1.mp3")

	opening = random.choice(opening_lines)
	closing = random.choice(closing_lines)

	speaktext(opening)

	turns.append({"speaker": AGENT, "text": opening})


    # load the personality of Plantony
	prompt_text = open("/Users/ya/LLMs/PLLantoid/v5/plantony.txt").read().strip()


    # obtain audio from the microphone

	while( len(turns) < maxturns ):

		listenSpeech()
		usertext = recoSpeech(AUDIO_FILE)
	
		if(usertext): print("**** I HEARD: " + usertext)
		else: usertext = "Hmmmm..."

		turns.append({"speaker": USER, "text": usertext})
                
		updated_prompt = updateprompt(turns)
	
		messages = gptmagic(updated_prompt)
		speaktext(messages)

	speaktext(closing);
	playsound("/Users/ya/LLMs/PLLantoid/v5/samples/outro1.mp3");

	os.system("python3 /Users/ya/LLMs/PLLantoid/v5/plantony2-goerli.py");


# device_info = audio.get_device_info_by_index(device_index)
