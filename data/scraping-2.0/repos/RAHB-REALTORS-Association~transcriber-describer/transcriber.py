import openai
from subsai.main import SubsAI

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as file:
        transcription = openai.Audio.transcribe("whisper-1", file)
    return transcription.text

def transcribe_audio_local(audio_file_path):
    subs_ai = SubsAI()
    model = subs_ai.create_model('openai/whisper', {'model_type': 'base'})
    transcription = subs_ai.transcribe(audio_file_path, model)
    return transcription
