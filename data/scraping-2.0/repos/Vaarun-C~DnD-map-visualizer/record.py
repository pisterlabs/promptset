import openai

openai.api_key = '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'

audio_file= open("./No Resolve - Get Me Out.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)

print(transcript)