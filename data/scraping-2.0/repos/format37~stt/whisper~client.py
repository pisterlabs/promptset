import openai
import json
import logging
import pickle
from pydub import AudioSegment


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def recognize_whisper(audio_path, api_key):
    logging.info('Starting the transcription process...')
    
    # OpenAI's Python package uses environment variables for API keys,
    # but since you're reading from a file, we'll set it directly.
    openai.api_key = api_key

    # Load the audio file
    logging.info(f'Loading audio file from {audio_path}...')
    with open(audio_path, "rb") as audio_file:
        # Transcribe the audio
        logging.info('Transcribing the audio...')
        response = openai.Audio.transcribe(
            file=audio_file,
            model="whisper-1",
            response_format="text",
            language="ru"
        )

        # Pickle the response for debugging
        with open("response.pkl", "wb") as f:
            pickle.dump(response, f)
        logging.info('Response pickled for debugging.')

    # Directly return the response as it's already a string
    logging.info('Transcription completed.')
    return response


def main():
    # Read API key from key.txt
    logging.info('Reading API key...')
    with open("key.txt", "r") as f:
        api_key = f.read().strip()

    audio_path = "audio.mp3"

    # Perform the transcription
    transcription = recognize_whisper(audio_path, api_key)

    # Save the transcription to a file
    logging.info('Saving the transcription to a file...')
    with open("transcription.txt", "w") as f:
        f.write(transcription)
    logging.info('Transcription saved.')

if __name__ == "__main__":
    main()
