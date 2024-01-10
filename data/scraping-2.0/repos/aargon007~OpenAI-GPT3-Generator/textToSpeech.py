from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

openai_api_key = os.getenv("OPENAI_API_KEY")  # Get API key from environment variables
client = OpenAI(api_key=openai_api_key)  # Set API key for OpenAI client

# text to speech generation
def generate_text_to_speech(text):
    audio_folder = Path(__file__).parent / "audio"  # Define the audio folder path
    audio_folder.mkdir(parents=True, exist_ok=True)  # Create audio folder if it doesn't exist

    # Get the first two words of the input text
    first_two_words = ' '.join(text.split()[:2])

    # Define the speech file path with a dynamic title based on the first two words
    speech_file_path = audio_folder / f"{first_two_words.lower()}_speech.mp3"

    # Generate speech from text using OpenAI API
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text  # Use the input text parameter here
    )

    # Save the speech response to an MP3 file with the dynamic title
    response.stream_to_file(speech_file_path)

    return speech_file_path  # Return the path of the generated speech file

# Example text to convert to speech
input_text = "Experiment with different voices (alloy, echo, fable, onyx, nova, and shimmer) to find one that matches your desired tone and audience. The current voices are optimized for English."

# Generate speech and get the path of the saved file
generated_speech_path = generate_text_to_speech(input_text)
print(f"Speech generated and saved as '{generated_speech_path}'")