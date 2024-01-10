import openai

f = open('../api_key.txt')
api_key = f.read()
openai.api_key = api_key

audio_file = open('../data/audio/test.m4a', 'rb')

transcription = openai.Audio.transcribe('whisper-1', audio_file)

print(transcription['text'])