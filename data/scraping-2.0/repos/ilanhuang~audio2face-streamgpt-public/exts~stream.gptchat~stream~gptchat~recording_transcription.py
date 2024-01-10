#Stream-GPT
#GNU - GLP Licence
#Copyright (C) <year>  <Huang I Lan & Erks - Virtual Studio>
#This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#You should have received a copy of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import pyaudio
import wave
import keyboard
import time
from time import sleep
import openai
import datetime

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:

        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)
        
def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")

def record_client_voice(output_filename, recording_status):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    frames = []

    p = pyaudio.PyAudio()
    stream = None

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        start_time = time.time()
        min_duration = 0.1

        while recording_status() or time.time() - start_time < min_duration:
            data = stream.read(CHUNK)
            frames.append(data)

    except Exception as e:
        print(f"Error while recording audio: {e}")
        
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()

        p.terminate()

        wf = wave.open(output_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    return output_filename
    
def transcribe_audio_to_text(file_path):
    with open(file_path, 'rb') as audio_file:
        transcript_response = openai.Audio.transcribe("whisper-1", audio_file)     

    return transcript_response["text"]