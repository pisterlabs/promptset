import openai
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the file to transcribe
file_path = "./offon.mp3"

# Define function to convert Unicode to readable format and save to file


def save_transcription(transcription):
    # Extract the text from the transcription object
    text = transcription["text"]

    # Convert Unicode characters to readable format
    # text_readable = text.encode('ascii', 'ignore').decode('utf-8')

    # Save the text to a file
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(text)

    # Print a message indicating that the file has been saved
    print("Transcription saved to file 'transcription.txt'")


# Start the timer
start_time = time.time()

# Load the audio file as a file-like object
with open(file_path, "rb") as f:
    # Transcribe the audio using the OpenAI API
    transcription = openai.Audio.transcribe("whisper-1", f)

# End the timer
end_time = time.time()

# Save the transcription to a file
save_transcription(transcription)

print(transcription)

# Calculate the execution time
execution_time = end_time - start_time

# Print the execution time
print("Execution time:", execution_time)
