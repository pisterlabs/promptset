import os
import sys
import time
import openai
import tempfile
import requests
import urllib.request
import speech_recognition as sr
from pydub import AudioSegment
from octorest import OctoRest
import nltk
from io import BytesIO
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from urllib3.exceptions import InsecureRequestWarning
from bs4 import BeautifulSoup
from thingiverse import Thingiverse
if sys.version_info >= (3, 6):
    import zipfile
else:
    import zipfile36 as zipfile

import pygcode
import io
from os import path
import asyncio
from stl import mesh
import py3d
from pydantic import BaseModel
import pydantic as pyd
from pygcode import Line
import time
import spacy

requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

octoprint_address = '{octoprint url)'
api_key = '{octoprint api key}'
thingverse_api = '{thingiverse api key}'
openai.api_key = '{openai api key}'
access_token ="{thingiverse access token}"

BUFFER_DURATION = 5 
TRIGGER_PHRASE = 'hey jarvis'
REQUEST_DELAY = 1

trigger_word = "start"
nlp = spacy.load("en_core_web_sm")

pp = "Fabricator"

def make_client(url, apikey):
     try:
         client = OctoRest(url=url, apikey=apikey)
         return client
     except ConnectionError as ex:
         print(ex)

def wait_for_trigger_word(trigger):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")

        while True:
            audio_data = recognizer.record(source, duration=BUFFER_DURATION)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
                tmp_audio_file.write(audio_data.get_wav_data())

            transcription = transcribe_whisper(tmp_audio_file.name)
            print(f"Transcription: {transcription}")

            os.remove(tmp_audio_file.name)

            if trigger.lower() in transcription.lower():
                print(f"Trigger phrase '{TRIGGER_PHRASE}' detected!")
                break
         
def add_suffix(strings):
    suffixed = [s + '.gco' for s in strings]
    return suffixed

def subtract_suffix(strings):
    subtracted = [s[:-4] if s.endswith('.stl') else s for s in strings]
    return subtracted


def sort_files_by_size(file_paths):
    file_sizes = [(path, os.path.getsize(path)) for path in file_paths]
    sorted_sizes = sorted(file_sizes, key=lambda x: x[1])
    sorted_paths = [path for path, _ in sorted_sizes]
    return sorted_paths
         


def download_stl(search_term, octoprint_address, api_key, access_token):
    cwd = os.getcwd()
    term = search_term
    trigger = "start"
    octo = make_client(url=octoprint_address, apikey=api_key)
    thingy = Thingiverse(access_token=access_token)
    search_results = thingy.search_term(term)
    print(search_results)
    
    id_num = search_results["hits"][0]['id']
    print(id_num)
    thing_url = "https://www.thingiverse.com/thing:" + str(id_num) + "/zip"
    response = requests.get(thing_url)

    zipf = zipfile.ZipFile(io.BytesIO(response.content))
    os.chdir('models')
    if not os.path.exists(str(term)):
        os.mkdir(str(term))
        os.chdir(str(term))

    else:
        os.chdir(str(term)) 
        
    path_str = 'C:/Users/Manav/Desktop/VOICEPRINT/models/' + str(term)
    zipf.extractall(path_str)
    os.chdir("files")
    path_str += "/files/"
    gcode_files = []
    stl_files = [f for f in os.listdir(path_str) if f.endswith(".stl")]
    sorted_files = sort_files_by_size(stl_files)
    print(sorted_files)
    file_names = subtract_suffix(sorted_files)
    print(file_names)
    files = add_suffix(file_names)
    gcode_paths = []
    for i in range(0, len(stl_files), 1):
        upload = octo.upload(stl_files[i])
        file_url = upload['files']['local']['name']
        filename = upload['files']['local']['refs']['resource']

        octo.slice(file_url, print=True, printer_profile=pp)

        print(octo.state())

        time.sleep(30)
        

        if octo.state()== "Operational":
            print("Say the command to start printing next slice")
            wait_for_trigger_word(trigger)
            print("command next recieved")
            if octo.state() != "Printing":
                octo.home()
                octo.select(files[i])
                resp = octo.start()

        while octo.state() == "Printing":
            print("waiting...")
            time.sleep(200)
             
    os.chdir('..')
    os.chdir('..')
    os.chdir('..')
    return(stl_files)


def get_noun_phrase_complement(sentence):
    doc = nlp(sentence)
    noun_phrase_complements = []
    for token in doc:
        if token.head.pos_ == "VERB" and token.dep_ == "dobj":
            noun_phrase = " ".join([child.text for child in token.subtree])
            noun_phrase_complements.append(noun_phrase)
    return noun_phrase_complements

def transcribe_whisper(audio_file_path):
    with open(audio_file_path, "rb") as audio_file: 
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)
    return transcript["text"]

def listen_and_transcribe():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")

        while True:
            audio_data = recognizer.record(source, duration=BUFFER_DURATION)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
                temp_audio_file.write(audio_data.get_wav_data())

            transcription = transcribe_whisper(temp_audio_file.name)
            print(f"Transcription: {transcription}")

            os.remove(temp_audio_file.name)

            if TRIGGER_PHRASE.lower() in transcription.lower():
                print(f"Trigger phrase '{TRIGGER_PHRASE}' detected!")
                sentence = transcription.lower()
                nouns = get_noun_phrase_complement(sentence)
                print(nouns)
                files = download_stl(nouns, octoprint_address, api_key, access_token)

                
if __name__ == "__main__":
    listen_and_transcribe()
