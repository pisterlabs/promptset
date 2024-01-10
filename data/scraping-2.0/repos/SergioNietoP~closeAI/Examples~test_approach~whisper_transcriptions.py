import openai
import time
import os

# Start the timer
start_time = time.time()

# Set the API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Open the audio file and transcribe it
with open("audios/audio_answer/answer_english_2.mp3", "rb") as audio_file:
    transcript = openai.Audio.transcribe(
        file = audio_file,
        model = "whisper-1",
        response_format="text"
    )

# Print the transcript
print(transcript)


# Calculate the elapsed time
elapsed_time = time.time() - start_time
print("Elapsed Time:", elapsed_time, "seconds")