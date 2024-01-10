import streamlit as st
import speech_recognition as sr
import whisper
import pyaudio
import numpy as np
import os
import openai
from secret_key import openai_key
openai.api_key = openai_key

st.title('StreamVoice Analytics')

model = whisper.load_model("base")
language = "en"

CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

r = sr.Recognizer()

temp_transcriptions = ""

# Create a temporary directory for uploaded audio
TEMP_DIR = "temp_text"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

temp_text_path = os.path.abspath(os.path.join(TEMP_DIR, "transcription.txt"))

def read_from_txt():
        with open(temp_text_path, 'r') as file:
            return file.read()

def save_to_txt(temp_transcriptions):
                with open(temp_text_path, 'w') as file:
                    file.write(temp_transcriptions)

# create 3 buttons in a row
col1, col2, col3 = st.columns(3)

# Place a button in each column
start = col1.button('Voice On')
stop = col2.button('Voice Off')
response = col3.button('Ask AI')

if start:
    st.write('Recording...')
    audio_data = []
    while not stop:
        # Continuously read from the audio stream
        buffer = stream.read(CHUNK_SIZE)
        audio_data.append(buffer)
        
        if len(audio_data) * CHUNK_SIZE / RATE > 5:  # transcribe every 5 seconds
            audio_chunk = b''.join(audio_data)
            audio = sr.AudioData(audio_chunk, RATE, 2)
            
            # Convert AudioData to numpy array
            audio_np = np.frombuffer(audio.frame_data, dtype=np.int16).astype(np.float32) / 32768
            
            try:
                transcription = whisper.transcribe(model=model, language=language ,audio=audio_np, fp16=False)
                temp_transcriptions += (transcription["text"])
                st.write(transcription["text"])
                save_to_txt(temp_transcriptions)
            except Exception as e:
                st.error(f"Error during transcription: {e}")
            
            audio_data = []  # Clear buffer
            # os.remove(temp_text_path)


if stop:
    transcription = read_from_txt()
    st.write("Recording stopped!...")
    st.write("Transcription: " + transcription)

if response:
    transcription = read_from_txt()
    st.write("Response: " + transcription)
    prompt = transcription

    def get_completion(prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
        )
        return response.choices[0].message["content"]

    response = get_completion(prompt)
    st.write(response)
    
stream.stop_stream()
stream.close()
p.terminate()
