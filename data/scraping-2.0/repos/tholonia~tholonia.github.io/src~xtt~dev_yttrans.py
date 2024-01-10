#!/bin/env python
"""
Options:
    -h, --help          show help
    -b, --base
    -s, --source        filename
    -v, --verbose       default: OFF
"""
import sys, getopt, os
from pprint import pprint
import subprocess
import whisper
#! for ASNI colors outout
import lib_ollama as lo

from colorama import init, Fore, Back
init()

from pytube import YouTube


#! ---------------------------------------------------------------------------
#! Set default values
source = False
verbose = False
speak = False
video_file = False
fromlang = "Spanish"
tolkang = "English"
video_file =  "/home/jw/Videos/vid.mp4"



argv = sys.argv[1:]
try:
    opts, args = getopt.getopt(argv,"hf:s:vS:F:T:",
                               ["help","video_file=","source=","verbose","speak","fromlang","tolang"])
except Exception as e:
    print(str(e))

#! set defaults
for opt, arg in opts:
    if opt in ("-h", "--help"): showhelp()
    if opt in ("-f", "--video_file"): video_file = arg
    if opt in ("-s", "--source"): source = arg
    if opt in ("-v", "--verbose"): verbose = True
    if opt in ("-S", "--speak"): speak = True
    if opt in ("-F", "--fromlang"): fromlang = arg
    if opt in ("-T", "--tolang"): tolang = arg

#^ ---------------------------------------------------------------------------

from openai import OpenAI


client = OpenAI(api_key="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")

# audio_file= open(video_file, "rb"),
transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=open(video_file,"rb")

)
# api_key=os.environ.get("OPENAI_API_KEY")


# API_KEY = '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'
# model_id = 'whisper-1'
# language = "en"
# audio_file_path =video_file
# audio_file = open(audio_file_path, 'rb')
#

# response = openai.Audio.transcribe(
#     api_key=API_KEY,
#     model=model_id,
#     file=audio_file,
#     language='en'
# )
transcription_text = transcript.text
print(transcription_text)

