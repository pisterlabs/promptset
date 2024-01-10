import openai

openai.api_key = ''
audio_file = open('output.mp3', 'rb')

transcript = openai.Audio.transcribe('whisper-1', audio_file)
print(transcript['text'])