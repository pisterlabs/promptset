
import os
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import torch
import numpy as np
import openai
from dotenv import load_dotenv
load_dotenv()
import pyperclip

def record(duration):
    fs = 44100  # this is the frequency sampling; also: 4999, 64000
    seconds = duration  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    print("Starting: Speak now!")
    sd.wait()  # Wait until recording is finished
    print("recording finished")
    write('output.mp3', fs, myrecording)  # Save as MP3 file

def transscribe():
    torch.cuda.is_available()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = whisper.load_model("base", device=DEVICE)
    print(
        f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
        f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
    )

    audio = whisper.load_audio('output.mp3')
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions(language="en",    )
    result = whisper.decode(model, mel, options)
    print(result.text)

    result = model.transcribe('output.mp3')
    print(result["text"])
    return result["text"]

def generate_mail(text):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=f"Write a kind email for this: I can not come to work today because im really sick\n\nHi there,\n\nI'm sorry for the short notice, but I won't be able to come in to work today. I'm really sick and need to rest. I'll be back to work tomorrow. Hope you all have a wonderful day.\n\nThanks,\n\n[Your Name]\n\n\nWrite a kind email for this: {text}",
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text

def main():
    record(8) # in seconds
    text = transscribe()
    email = generate_mail(text)
    print(email)
    pyperclip.copy(email)

if __name__ == "__main__":
    main()