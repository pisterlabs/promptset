import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

audio_file= open("sample/whisper_english.mp4", "rb")
# transcript = openai.Audio.translate("whisper-1", audio_file)
transcript = openai.Audio.translate(
    model="whisper-1",
    file=audio_file,
    prompt="This youtuber is CodeMKE")
print(transcript['text'])