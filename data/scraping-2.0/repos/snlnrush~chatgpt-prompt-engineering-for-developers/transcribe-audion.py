from openai import OpenAI

# Set your API key
with open('api_key.txt', 'r') as file:
    api_key = file.readline().strip()

client = OpenAI(api_key=api_key)

# Open the openai-audio.mp3 file
audio_file = open('audio_example.ogg', 'rb')

# Create a transcript from the audio file
response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

# Extract and print the transcript text
print("Text from audio:\n--- Beginning\n", response.text, '\n--- End')
