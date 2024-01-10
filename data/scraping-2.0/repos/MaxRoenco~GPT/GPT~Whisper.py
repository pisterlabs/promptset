import openai

KEY = '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'

file = open('bg.mp3', 'rb')

result = openai.Audio.translate(
    api_key=KEY,
    model='whisper-1',
    file=file,
    response_format='text'
)

print(result)

