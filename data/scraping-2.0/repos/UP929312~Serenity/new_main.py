import json
import wave
import subprocess
from typing import Iterator

import openai
import pyaudio
import keyboard
import faster_whisper

from elevenlabs import set_api_key, voices, generate

with open("keys/openai_key.txt") as f:
    openai.api_key = f.read().strip()

with open("keys/elevenlabs.txt") as f:
    set_api_key(f.read().strip())

with open("assets/files/prompts.json") as f:
    prompts = json.load(f)


voice = [x for x in voices() if x.name == "Dorothy"][0]

model, answer, history = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device='cpu'), "", []

def generate_next_response(messages):
    global answer
    answer = ""
    for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")):  # type: ignore
            answer += text_chunk
            print(text_chunk, end="", flush=True) 
            yield text_chunk

def custom_stream(audio_stream: Iterator[bytes]) -> bytes:
    mpv_command = ["C:/Users/Ben/Desktop/mpv_base/mpv.exe", "--no-cache", "--no-terminal", "--", "fd://0"]
    mpv_process = subprocess.Popen(
        mpv_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    audio = b""

    for chunk in audio_stream:
        if chunk is not None:
            mpv_process.stdin.write(chunk)  # type: ignore
            mpv_process.stdin.flush()  # type: ignore
            audio += chunk

    if mpv_process.stdin:
        mpv_process.stdin.close()
    mpv_process.wait()
    return audio

while True:
    # Wait until user presses space bar
    print("\n\nTap space when you're ready. ", end="", flush=True)
    keyboard.wait('space')
    while keyboard.is_pressed('space'): pass

    # Record from microphone until user presses space bar again
    print("I'm all ears. Tap space when you're done.\n")
    audio, frames = pyaudio.PyAudio(), []
    audio_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    while not keyboard.is_pressed('space'):
        frames.append(audio_stream.read(512))
    audio_stream.stop_stream(), audio_stream.close(), audio.terminate()    # type: ignore

    # Transcribe recording using whisper
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))
    user_text = " ".join(seg.text for seg in model.transcribe("voice_record.wav", language="en")[0])
    print(f'>>>{user_text}\n<<< ', end="", flush=True)
    history.append({'role': 'user', 'content': user_text})

    # Generate and stream output
    generator = generate_next_response([{"role": "system", "content": prompts["initial"]}] + history[-10:])
    custom_stream(generate(text=generator, voice=voice, model="eleven_monolingual_v1", stream=True))  # type: ignore
    history.append({'role': 'assistant', 'content': answer})