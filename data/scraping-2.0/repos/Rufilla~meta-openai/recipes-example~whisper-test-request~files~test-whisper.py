import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

audio_file= open("/usr/bin/audio-example.mp3", "rb")         
transcript = openai.Audio.transcribe("whisper-1", audio_file)
                                                                                                     
# print the transcription result                             
print(transcript)
