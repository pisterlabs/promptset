import os
import openai

openai.api_key = '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'

audio_file= open("sample-0.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript)