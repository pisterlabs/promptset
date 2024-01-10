import openai
from env.env import config

def transcribe_audio():
    audio_file= open("media/voice_note.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript