import modal
import os
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import tempfile
import json

stub = modal.Stub("stt", image=modal.Image.debian_slim().pip_install("openai~=0.27.0", "python-multipart"))


@stub.webhook(method="POST", secret=modal.Secret.from_name("my-openai-secret") )
def stt(file: UploadFile = File(...)):
    import openai
    # print(file.filename)
    # print(file.content_type)
    # print(file.file)
    # print(file.file.name)
    # print(os.getcwd())
    # print(os.listdir())
    with open("new_file.webm", "wb+") as new_file:
        new_file.write(file.file.read())
    audio_file= open("new_file.webm", "rb")
    response = openai.Audio.transcribe("whisper-1", audio_file)
    os.remove("new_file.webm")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Your job is to reproduce the transcript as faithfully as possible. Do not change too much or add in any of your own thoughts to the transcript."},
            {"role": "user", "content": "I am going to give you a transcript of an audio file and I want you to remove filler words and fix obvious mistakes in transcription. Your job is to reproduce the transcript as faithfully as possible. Do not change too much or add in any of your own thoughts to the transcript. Return to me only your edited version of the transcript. Here is the transcript: " + response.text}
        ]
    )
    response = response["choices"][0]["message"]["content"]
    print(response)
    return {"text": response}

