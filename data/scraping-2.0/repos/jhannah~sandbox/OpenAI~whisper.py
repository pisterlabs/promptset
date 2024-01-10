# https://platform.openai.com/docs/tutorials/meeting-minutes
# pip3 install openai

# Gotta split into ~10MB files because 25MB is the theoretical max
# ffmpeg -i /Users/jhannah/Dropbox/Public/jay_flaunts/043.mp3 -f segment -segment_time 1200 -c copy ./out%03d.mp3
# In vim: gq to add hard word wrapping

import openai

def transcribe_audio(audio_file_path):
  with open(audio_file_path, 'rb') as audio_file:
    transcription = openai.Audio.transcribe("whisper-1", audio_file)
  return transcription['text']

text = transcribe_audio("out003.mp3")
print(text)


