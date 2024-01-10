import sys
import time
from openai import OpenAI
from colorama import Fore
import numpy as np
import threading
from queue import Queue
from threading import Thread
from colorama import Fore, Style
from collections import deque
sys.path.append('C:\SOURCE\GERRY\openai-whisper-test1')
import globals

# audio processing - conversation does the conversion to wave as well
from operations.audio_input_conversation import record_audio_conversation
#from operations.audio_input import convert_array_to_wave

# AI stuff
from operations.speech_to_text_conversation import speech_to_text_conversation
from operations.text_to_speech_conversation import text_to_speech_conversation

# Add a variable to keep track of the last processed index
# This is for a technique to feed one second of the previous audio to the next transcription to avoid
# missing words that might get cut off, let's see if it works!
last_processed_index = 0

# i guess we set globals.global_tts_input_string here
globals.global_tts_input_string = ""

### THREADS ###

# Create a thread for the speech to text function
speech_to_text_conversation_thread = threading.Thread(target=speech_to_text_conversation)
speech_to_text_conversation_thread.start()

# Create a thread for the text to speech function
text_to_speech_embedded_thread = threading.Thread(target=text_to_speech_conversation)
text_to_speech_embedded_thread.start()

# Create a thread for the record_audio function
audio_input_conversation_thread = threading.Thread(target=record_audio_conversation)
audio_input_conversation_thread.start()

# Create a thread for the convert_array_to_wave function
#convert_thread = threading.Thread(target=convert_array_to_wave)
#convert_thread.start()





























# Read API key from a file
with open('secret_apikey.txt', 'r') as file:
    api_key = file.read().strip()

client = OpenAI(api_key=api_key)

# specify the OpenAI model to use:
# https://platform.openai.com/docs/models/gpt-3-5
# 16k token limit on "gpt-3.5-turbo-16k-0613"
# 16k token limit on "gpt-3.5-turbo-1106" (newest release as of dec-1-2023)
gpt_model = "gpt-3.5-turbo-1106"

# Our first AI 'assistant' role and its speciality
openai_specialization = "Just a regular dude"
# The base premise of what we are trying to do
base_premise = "You will just behave like a regular dude."

####
while True:
    if not globals.global_tts_input_string == "THREADPAUSE" or globals.global_tts_input_string is None:
        #print("globals.global_tts_input_string is not THREADPAUSE, waiting 1 second...")
        time.sleep(0.2)  # check every 200ms":
        chat = client.chat.completions.create(
          model=gpt_model,
          max_tokens=512,
          messages=[
            {"role": "system", "content": openai_specialization + base_premise,
             "role": "user", "content": globals.global_tts_input_string
            }])

        chat_response_text = chat.choices[0].message.content
        print(f"{Fore.CYAN}{chat_response_text}{Fore.RESET}\n")
        chat_response_text = globals.global_tts_input_string

