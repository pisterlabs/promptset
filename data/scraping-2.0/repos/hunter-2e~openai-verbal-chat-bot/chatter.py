#TEXT TO SPEECH
from gtts import gTTS
import os

#RESPONSE AI
import openai

#SPEECH TO TEXT
import whisper
import wave
import pyaudio

model = whisper.load_model("medium")
openai.api_key = "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"

audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

frames = []

try:
    while True:
        data = stream.read(1024)
        frames.append(data)
except KeyboardInterrupt:
    pass

stream.start_stream()
stream.close()
audio.terminate()

sound_file = wave.open("human.wav", "wb")
sound_file.setnchannels(1)
sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
sound_file.setframerate(44100)
sound_file.writeframes(b''.join(frames))
sound_file.close()

prompt = model.transcribe("human.wav")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Why sky blue.",
  temperature=0.9,
  max_tokens=150,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.6,
  stop=[" Human:", " AI:"]
)

myobj = gTTS(text=response["choices"][0]["text"], lang='en', slow=False)

#SAVE THEN PLAY MP3
myobj.save("bot.mp3")
os.system("bot.mp3")

print(response["choices"][0]["text"])