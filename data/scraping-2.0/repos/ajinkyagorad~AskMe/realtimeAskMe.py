import numpy as np
import queue
import threading
import time
import openai
import whisper
import soundfile as sf
import io
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from time import time
import sounddevice as sd
import os
from playsound import playsound
import simpleaudio
import struct
import boto3
from textblob import TextBlob
import cv2
import pyttsx3
from io import BytesIO
import requests
from PIL import Image
# pyttsx3


# AWS access keys (incomplete and are temporary) hint AKvSAG
aws_access_key_id = 'IA3ZOK3GYW2647Q54M'
aws_secret_access_key = 'gKMsBFkYBzlJS/XKqC8idobl6b/jJKkeiX8xmk'
# OpenAI api key
openai.api_key = 'sk-aalWvYCWi0oiYffnsk5WT3BlbkFJn2snshVgPGXwlG3Hn9'  

# AWS region name
region_name = 'eu-central-1'

# Create a client session
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

# Create a Polly client
polly_client = session.client('polly')
comprehend = session.client('comprehend')

q = queue.Queue()
samplerate = 44100
recording_duration = 50 # timeout (sec)
inactive_time_limit = .5 # when person pauses for this time or more (sec)
recording_blocks = []
dirpath = r''

valid_language_codes = ['en-US', 'en-IN', 'es-MX', 'en-ZA', 'tr-TR', 'ru-RU', 'ro-RO', 'pt-PT', 'pl-PL', 'nl-NL', 'it-IT', 'is-IS', 'fr-FR', 'es-ES', 'de-DE', 'yue-CN', 'ko-KR', 'en-NZ', 'en-GB-WLS', 'hi-IN', 'arb', 'cy-GB', 'cmn-CN', 'da-DK', 'en-AU', 'pt-BR', 'nb-NO', 'sv-SE', 'ja-JP', 'es-US', 'ca-ES', 'fr-CA', 'en-GB', 'de-AT']
def draw_image(prompt):
    
    response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="512x512"
            )
    print('submitted request.. waiting')
    image_url = response['data'][0]['url']
    response = requests.get(image_url)
    print('response at: ', image_url)
    # Open the response content as an image using Pillow
    image = Image.open(BytesIO(response.content))
    #cv2.namedWindow('imgen',  cv2.WINDOW_KEEPRATIO )
    cv2.imshow('imgen',np.array(image))
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def audio_callback(indata, frames, time, status):
    q.put(indata.copy())
def text_to_speech_offline(text):
    # Initialize the engine
    engine = pyttsx3.init()

    # Set the voice property to a female voice
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id) # index 1 corresponds to a female voice
    engine.say(text)

    # Run the engine and wait until speech is complete
    engine.runAndWait()
    print('¤')
def text_to_speech_aws(text):
    response = comprehend.detect_dominant_language(Text=text)

    # extract language code
    language_code = response['Languages'][0]['LanguageCode']
    language_code = language_code+'-'+language_code.upper()
    

    if language_code not in valid_language_codes:
        language_code = valid_language_codes[0] 

    # Generate an MP3 file using Polly
    print('detected as:', language_code)
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId='Joanna',
        LanguageCode=language_code
    )
    ofile = os.path.join(dirpath, 'output.mp3')
    # Save the MP3 file to disk
    with open(ofile, 'wb') as file:
        file.write(response['AudioStream'].read())
    
    playsound('output.mp3')
    '''
    sound = AudioSegment.from_mp3(ofile)
    sound.export('output.wav', format="wav")
    obj = simpleaudio.WaveObject.from_wave_file('output.wav')
    pobj = obj.play()
    pobj.wait_done()
    '''

def text_to_speech(text):
    tts = gTTS(text, lang="en", slow=False)
    #fp = io.BytesIO()
    ofile = os.path.join(dirpath, 'output.mp3')
    tts.save(ofile)
    sound = AudioSegment.from_mp3(ofile)
    sound.export('output.wav', format="wav")
    obj = simpleaudio.WaveObject.from_wave_file('output.wav')
    pobj = obj.play()
    pobj.wait_done()
    #print('filepath: ',os.path.join(dirpath, 'output.mp3')) 
    
    #audio_file = AudioSegment.from_file('output.mp3', format="mp3")
    #play(audio_file)
def process_audio():
    while True:
        recording_blocks = []
        print('recording...')
        last_active_time = time()
        inactive_time = 0
        start_time = time()
        while True:
            audio_data = q.get()[:,0]
            if np.max(audio_data)<0.01:
               inactive_time = time()-last_active_time
            else:
                last_active_time = time()
            recording_blocks.append(audio_data)
            if inactive_time>inactive_time_limit or len(recording_blocks) * audio_data.shape[0] >= samplerate * recording_duration:
                break
        print('done')
        audio_data_concat = np.concatenate(recording_blocks, axis=0)
        # only proceed if at least 1 second of audio is present and there is at least 50% audio else redo
        if time()-start_time<1:
            print('too short')
            continue
        val=np.sum(audio_data_concat>0.005)/len(audio_data_concat)
        if val<.05:
            print('too little audio :', val)
            continue

        sf.write(os.path.join(dirpath, 'input.wav'), audio_data_concat, samplerate)
        with open(os.path.join(dirpath, 'input.wav'), 'rb') as f:
            transcript = openai.Audio.transcribe("whisper-1", f)['text']
            print('Input: ',transcript)
        if transcript.split(' ')[0].upper() == 'DRAW':
            draw_image(transcript)
        elif transcript:
            response = openai.Completion.create(engine="text-davinci-003", prompt=transcript, max_tokens=50, n=1, stop='None', temperature=0.7)
            message = response.choices[0].text.strip()
            if message:
                print("Response:", message)
                #text_to_speech_aws(message)
                text_to_speech_offline(message)
                

stream = sd.InputStream(device = 0, callback=audio_callback)
#outstream=sd.OutputStream(samplerate=samplerate)
stream.start()
#outstream.start()

processing_thread = threading.Thread(target=process_audio)
processing_thread.start()

processing_thread.join()

while True:
    # Keep the main thread running until the user presses the 'q' key
    if input() == 'q':
        break
stream.stop()
stream.close()
