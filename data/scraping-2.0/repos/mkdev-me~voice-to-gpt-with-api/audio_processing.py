import torch
import openai
import pyaudio
import wave
import time
import openai
import audioop
import os
import subprocess

openai.api_key = os.environ.get("OPENAI_API_KEY")

def transcribe_audio(file):
    file_root, _ = os.path.splitext(file)
    output_file = file_root + ".mp3"
    command = ['ffmpeg', '-i', file, output_file]
    subprocess.run(command, check=True)
    audio_file= open(output_file, "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)

    print(transcript)
    return transcript

def ask_gpt(prompt, max_tokens=300):
    prompt = f"Conversación con un asistente AI:\n{prompt}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.2,
    )

    print (response.choices[0].text.strip())
    return response.choices[0].text.strip()

