from openai import OpenAI

# Set your API key
with open('api_key.txt', 'r') as file:
    api_key = file.readline().strip()

client = OpenAI(api_key=api_key)

# Open the audio file
audio_file = open('audio_example.ogg', 'rb')

# Write an appropriate prompt to help the model
prompt = "The audio relates to a broken laptop and the broken web-sites."

# Create a translation from the audio file
response = client.audio.translations.create(
    model='whisper-1',
    file=audio_file,
    prompt=prompt
    )

# Extract and print the transcript text
print("Translated text from audio:\n--- Beginning\n", response.text, '\n--- End')
