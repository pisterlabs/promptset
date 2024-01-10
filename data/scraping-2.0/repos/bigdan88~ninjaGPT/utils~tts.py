import openai
from pathlib import Path
from playsound import playsound
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_api_key(file_path):
    """
    Reads the API key from a file.

    Args:
        file_path (str): Path to the file containing the API key.

    Returns:
        str: The API key.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read().strip()
    except IOError:
        logging.error("Unable to read API key. Check if the credentials file exists and is readable.")
        return None

# Read API key from credentials file
api_key = read_api_key('credentials.txt')
if not api_key:
    logging.critical("API key not found. Exiting.")
    exit(1)

# Initialize OpenAI client with the API key
client = openai.Client(api_key=api_key)

def text_to_speech(text):
    """
    Converts the given text to speech using OpenAI's text-to-speech API.

    Args:
        text (str): The text to convert to speech.

    Returns:
        str: The path to the saved audio file.
    """
    # Path for the speech file
    speech_file_path = Path(__file__).parent / "speech.mp3"

    try:
        # Create the audio file using OpenAI's text-to-speech API
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )

        # Save the audio file
        response.stream_to_file(speech_file_path)

        # Play the audio file
        playsound(str(speech_file_path))
        logging.info(f"Speech successfully generated and saved to {speech_file_path}")
    except Exception as e:
        logging.error("Failed to convert text to speech: " + str(e))

if __name__ == "__main__":
    text_to_speech("Hello world!")  # Replace with your actual text
