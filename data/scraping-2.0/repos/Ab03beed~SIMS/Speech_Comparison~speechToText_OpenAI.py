import openai
import os
import time

# Set the API key
openai.api_key = os.getenv('GPT_API_KEY')

# Make sure the file path is correct
audio_file_path = 'C:/Users/abdah/OneDrive/سطح المكتب/test/microphone-results.wav'

total = 0

# Open the audio file outside the loop to avoid reopening it multiple times
with open(audio_file_path, 'rb') as audio_file:
    for i in range(10):
        audio_file.seek(0)  # Reset file pointer to the beginning
        startTime = time.time()
        text = openai.Audio.transcribe('whisper-1', audio_file)
        endTime = time.time()
        res = endTime - startTime
        total += res
        print(i, ": ", res)

meanValue =  total / 10

print("mean value is: ", meanValue)