import pyaudio
import wave
import os
from openai import OpenAI
from dotenv import load_dotenv
from playsound import playsound
import whisper
import keyboard
import csv
import time

def get_recording():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16_000, input=True, frames_per_buffer=1024)
    frames = []

    print("\nRecording, press 'esc' to stop.")
    while True:
        data = stream.read(1024)
        frames.append(data)
        if keyboard.is_pressed('esc'):
            break
    print("\nGenerating response...")
    return [frames, stream, audio]

def end_recording(frames, stream, audio):
    stream.stop_stream()
    stream.close()
    audio.terminate()
    sound_file = wave.open("output.wav", "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(16_000)
    sound_file.writeframes(b"".join(frames))
    sound_file.close()
    
def chat_answer(question, message_history):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Your name is Jarvis. You are a artificial intelligence assistant, very similar to the Jarvis from Iron Man. You will respond to questions in short detailed sentences"},
            {"role": "system", "content": f"Here is our conversation history, in 'question: answer' format: {message_history} The last key value in the dictionary is the most recent question you asked."},
            {"role": "assistant", "content": question}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def speech_to_text():
    whisp_model = whisper.load_model("base")
    result = whisp_model.transcribe("output.wav", fp16=False)
    result = result['text']
    return result

def text_to_speech(text):
    response = client.audio.speech.create(
      model="tts-1",
      voice="echo",
      input=text
    )
    response.stream_to_file("speech.mp3")
# ----------------------------------------------------------------------------------------------------------------------
#1 Record audio
load_dotenv()
message_history = {}
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(
    api_key=api_key
)

def run_jarvis():
    # Start Recording
    print("Press 'esc' to start recording. Press ctrl+C to quit.")
    while True:
        if keyboard.is_pressed('esc'):
            break
    recording = get_recording()
    # End Recording
    end_recording(recording[0], recording[1], recording[2])
    # Speech to Text
    result = speech_to_text()
    # Get Answer
    answer = chat_answer(result, message_history)
    message_history[result] = answer
    # Text to Speech
    text_to_speech(answer)
    # Play Answer
    playsound("speech.mp3")

print("Jarvis is online. Press ctrl+C to quit.")
try:
    while True:
        val = ""
        run_jarvis()
except KeyboardInterrupt:
    pass

# write message history to csv in the conversation_history folder, using current time and date as name. The first column is id, the second is the question, the third is the answer.
with open(f"conversation_history/memory-{time.strftime('%Y-%m-%d_%H.%M')}.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "question", "answer"])
    for i, (question, answer) in enumerate(message_history.items()):
        writer.writerow([i, question, answer])

print("\nJarvis is offline.")