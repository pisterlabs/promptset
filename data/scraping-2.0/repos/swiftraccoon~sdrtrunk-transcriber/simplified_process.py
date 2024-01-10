import openai
import logging
import requests
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="script_log.log",
    filemode="a",
)
logger = logging.getLogger()


# Configurations
RECORDINGS_DIR = "/home/YOUR_USER/SDRTrunk/recordings"
OPENAI_API_KEY = "YOUR_KEY_HERE"


def pyapi_transcribe_audio(file_path):
    openai.api_key = OPENAI_API_KEY
    audio_file = open(file_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return str(transcript)


def curl_transcribe_audio(file_path):
    # Define the endpoint and your API key
    url = "https://api.openai.com/v1/audio/transcriptions"
    api_key = OPENAI_API_KEY

    # Setup headers
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    # Open the file and setup files and data to be sent
    with open(file_path, "rb") as file:
        files = {
            "file": file,
        }
        data = {
            "model": "whisper-1",
            "response_format": "json",
            "temperature": "0",
            "language": "en",
        }

        # Make the POST request
        response = requests.post(url, headers=headers, files=files, data=data)

    # Print the response or handle as needed
    return str(response.json())


def process_file(file):
    logger.info(f"Processing file: {file}")
    if not file.endswith(".mp3"):
        return

    full_path = os.path.join(RECORDINGS_DIR, file)
    talkgroup_id = file.split("TO_")[1].split("_")[0]

    # Move the file based on talkgroup ID
    new_dir = os.path.join(RECORDINGS_DIR, talkgroup_id)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    new_path = os.path.join(new_dir, file)
    os.rename(full_path, new_path)

    # Transcribe the audio
    transcription = curl_transcribe_audio(new_path)
    logger.info(f"Transcribed text for {file}: {transcription}")

    # Write transcription to a text file
    try:
        logger.info(f"Starting to write to text file for {file}")
        with open(new_path.replace(".mp3", ".txt"), "w") as text_file:
            text_file.write(transcription)
    except Exception as e:
        logger.error(f"Error while writing to text file: {str(e)}")


def main():
    for file in os.listdir(RECORDINGS_DIR):
        process_file(file)


if __name__ == "__main__":
    main()
