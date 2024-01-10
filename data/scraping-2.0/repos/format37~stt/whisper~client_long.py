import logging
from pydub import AudioSegment
import os
import sys
import uuid
import openai


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_audio_ffmpeg(audio_path, chunk_length=10*60):
    """
    Splits the audio file into chunks using ffmpeg.
    Returns a list of paths to the chunks.
    """
    # Get the duration of the audio in seconds
    cmd_duration = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {audio_path}"
    duration = float(os.popen(cmd_duration).read())

    # Calculate number of chunks
    chunks_count = int(duration // chunk_length) + (1 if duration % chunk_length > 0 else 0)

    chunk_paths = []

    # Create 'tmp/' directory if it doesn't exist
    if not os.path.exists("tmp/"):
        os.makedirs("tmp/")

    for i in range(chunks_count):
        start_time = i * chunk_length
        # Unique filename for the chunk
        chunk_filename = f"tmp/{uuid.uuid4()}.mp3"
        # Use ffmpeg to extract a chunk of the audio
        # cmd_extract = f"ffmpeg -ss {start_time} -t {chunk_length} -i {audio_path} -acodec copy {chunk_filename}"
        cmd_extract = f"ffmpeg -ss {start_time} -t {chunk_length} -i {audio_path} -acodec libmp3lame {chunk_filename}"
        os.system(cmd_extract)

        chunk_paths.append(chunk_filename)

    return chunk_paths


def transcribe_chunk(audio_path, api_key):

    openai.api_key = api_key

    with open(audio_path, "rb") as audio_file:
        response = openai.Audio.transcribe(
        file=audio_file,  
        model="whisper-1",
        response_format="text"
        )

    return response


def recognize_whisper(audio_path, api_key):
    logging.info('Starting the transcription process...')

    # Split the audio into chunks
    chunk_paths = split_audio_ffmpeg(audio_path)

    full_text = ""
    
    for idx, chunk_path in enumerate(chunk_paths):
        logger.info(f"Processing chunk {idx+1} of {len(chunk_paths)}")

        # Load chunk into memory using pydub
        # chunk_audio = AudioSegment.from_file(chunk_path)

        # Transcribe chunk
        text = transcribe_chunk(chunk_path, api_key)
        full_text += text

        # Remove the temporary chunk file
        os.remove(chunk_path)

    return full_text


def main():

    # Read from sys cmd params
    if len(sys.argv) < 3:
        print("Usage: python client.py <audio_path> <text_path>")
        sys.exit(1)
    audio_path = sys.argv[1]
    text_path = sys.argv[2]

    # Read API key from key.txt
    logging.info('Reading API key...')
    with open("key.txt", "r") as f:
        api_key = f.read().strip()

    logger.info(f"Transcribing {audio_path} to {text_path}")

    # Perform the transcription
    transcription = recognize_whisper(audio_path, api_key)

    # Save the transcription to a file
    logging.info('Saving the transcription to a file...')
    with open(text_path, "w") as f:
        f.write(transcription)
    logging.info('Transcription saved.')


if __name__ == "__main__":
    main()
