import json
import os
import time
import wave
import re

import keyboard
import openai
import pyaudio
from dotenv import load_dotenv
from faster_whisper import WhisperModel

from modules.TTS import google_tts,silero_tts
from utils.voicevox import VoicevoxTTSPlugin
from modules.prompt import getPrompt
from modules.sendAudio import play_voice
from modules.translate import detect_google, translate_google, translate_deepL

# Load environment variables
load_dotenv()

# Define constants
OWNER_NAME = os.getenv('OWNER_NAME')
MIC_ID = int(os.getenv('MICROPHONE_ID'))
SPEAKER_ID = int(os.getenv('AUDIO_SPEAKER_ID'))
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_JSON_FILE')

# Create a dictionary to hold the message data
conversation = []
history = {"history": conversation}

# Load the Whisper ASR model
MODEL_SIZE = "small"
ASR_MODEL = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "input.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=MIC_ID)

    frames = []

    print("Press and hold Right Shift to record audio")
    while keyboard.is_pressed('RIGHT_SHIFT'):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Stopped recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    transcribe_audio(WAVE_OUTPUT_FILENAME)

def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            segments, _ = ASR_MODEL.transcribe(audio_file, beam_size=5)
            chat_now = " ".join(segment.text for segment in segments)
            print(f"You: {chat_now.strip()}")
    except openai.error.InvalidRequestError as e:
        print(f"Error transcribing audio: {e}")
        return

    conversation.append({'role': 'user', 'content': chat_now})
    openai_answer()

def openai_answer():
    total_characters = sum(len(d['content']) for d in conversation)

    while total_characters > 4000:
        try:
            conversation.pop(2)
            total_characters = sum(len(d['content']) for d in conversation)
        except Exception as e:
            print("Error removing old messages: {0}".format(e))

    unanswered_questions = [msg for msg in conversation if msg['role'] == 'user' and not msg.get('answered')]
    if not unanswered_questions:
        return

    with open("conversation.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    prompt = getPrompt()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        max_tokens=248,
        temperature=1,
        top_p=1
    )
    message = response['choices'][0]['message']['content']
    replies, situation = process_message(message)
    conversation.append({'role': 'assistant', 'content': replies})
    for msg in unanswered_questions:
        msg['answered'] = True

    # Save the conversation to the "conversation.json" file after appending the assistant's response
    with open("conversation.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    print(" " + situation)
    translate_text(replies)
    return situation
    

def process_message(message):
    # Find the text inside double quotes
    reply_pattern = r'"(.*?)"'
    reply = re.findall(reply_pattern, message)
    if not reply:
        replies = message
    else:
        replies = ". ".join(reply)
    # Replace the reply with an empty string and remove leading/trailing spaces
    situation = message

    return replies, situation

def translate_text(text):
    detect = detect_google(text)
    tts = translate_deepL(text, detect, "JA")
    #tts_id = translate_deepL(text, detect, "ID")
    #tts_en = translate_google(text, detect, "EN")

    plugin = VoicevoxTTSPlugin()
    plugin.tts(tts, SPEAKER_ID, 14, websocket_connection=None, download=False, save_locally=True)
    time.sleep(1)

    audio_device_ids = [SPEAKER_ID]
    play_voice(audio_device_ids)

    time.sleep(1)
    with open("output.txt", "w") as f:
        f.truncate(0)
    with open("chat.txt", "w") as f:
        f.truncate(0)