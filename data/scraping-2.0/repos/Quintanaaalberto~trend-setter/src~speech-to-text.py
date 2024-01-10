import openai
import os
import pyaudio
import keyboard
import numpy as np
from scipy.io import wavfile


class Recorder:
    def __init__(self, filename):
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.sample_rate = 44100
        self.chunk = int(0.03 * self.sample_rate)
        self.filename = filename
        self.START_KEY = 's'
        self.STOP_KEY = 'q'

    def record(self):
        recorded_data = []
        p = pyaudio.PyAudio()

        stream = p.open(format=self.audio_format, channels=self.channels,
                        rate=self.sample_rate, input=True,
                        frames_per_buffer=self.chunk)
        while True:
            data = stream.read(self.chunk)
            recorded_data.append(data)
            if keyboard.is_pressed(self.STOP_KEY):
                print("Stop recording")
                # stop and close the stream
                stream.stop_stream()
                stream.close()
                p.terminate()
                # convert recorded data to numpy array
                recorded_data = [np.frombuffer(frame, dtype=np.int16) for frame in recorded_data]
                wav = np.concatenate(recorded_data, axis=0)
                wavfile.write(self.filename, self.sample_rate, wav)
                print("You should have a wav file in the current directory")
                break

    def listen(self):
        print(f"Press `{self.START_KEY}` to start and `{self.STOP_KEY}` to quit!")
        while True:
            if keyboard.is_pressed(self.START_KEY):
                self.record()
                break


recorder = Recorder("mic.wav")  # name of output file
recorder.listen()

openai.api_key = os.getenv('OPENAI-API-KEY')
audio_file = open("mic.wav", "rb")

transcribed_audio = openai.Audio.transcribe(file=audio_file, model="whisper-1",
                                            response_format="srt")  # response format is txt by default


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]

    chatgpt_answer = openai.ChatCompletion.create(model=model, messages=messages, temperature=0)

    return chatgpt_answer.choices[0].message["content"]


num_speakers = int(input("Enter number of speakers"))

chatgpt_prompt = f"Transcribe the text line-by-line and show speaker tags knowing there are {num_speakers}: {transcribed_audio}"

formatted_transcription = get_completion(chatgpt_prompt)

print("\n\nGENERATED TRANSCRIPT\n")
print(formatted_transcription)

# Record audio in small instances and join it on output

# Intelligent Title generator that saves the file with appropriate name

# Generate transcript as txt file

# Identify if it is a conversation or not
# YES: Identify speakers
# NO: CHATGPT should automatically take care of building appropriate paragraphs.

# The recording should be done with the toggle of a button

# Add functions such as translate of summarize
