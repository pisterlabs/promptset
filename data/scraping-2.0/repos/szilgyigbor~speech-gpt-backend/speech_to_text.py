import openai
import os

openai.api_key = os.getenv('BotApiKey')


def transcribe_audio(incoming_audio):
    file_path = "temp_audio.m4a"
    incoming_audio.save(file_path)

    with open(file_path, 'rb') as audio_file:
        response = openai.Audio.transcribe("whisper-1", audio_file)

    transcription = response['text']

    os.remove(file_path)

    return transcription
