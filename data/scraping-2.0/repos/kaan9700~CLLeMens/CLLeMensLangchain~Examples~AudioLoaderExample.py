import os
import shutil
from pydub import AudioSegment
import openai
from dotenv import load_dotenv

# Laden Sie die Umgebungsvariablen aus der .env-Datei
load_dotenv()

# Setzen Sie Ihren OpenAI API-Schlüssel aus der Umgebungsvariable
openai.api_key = os.getenv("OPENAI_API_KEY")


def chunk_and_transcribe(audio_file_path, chunk_length_in_seconds=120):
    # Load audio file
    song = AudioSegment.from_mp3(audio_file_path)

    # Convert chunk length to milliseconds (PyDub uses milliseconds)
    chunk_length = chunk_length_in_seconds * 1000

    # Calculate number of chunks needed
    num_chunks = len(song) // chunk_length + 1  # +1 to account for any remaining part

    # Create cache directory if it doesn't exist
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Initialize an empty string to store the full transcription
    full_transcription = ""

    # Loop over the audio file, chunk by chunk
    for i in range(num_chunks):
        # Extract the chunk
        start_time = i * chunk_length
        end_time = (i + 1) * chunk_length
        chunk = song[start_time:end_time]

        # Export the chunk to a temporary file
        temp_file_name = os.path.join(cache_dir, f"temp_chunk_{i}.mp3")
        chunk.export(temp_file_name, format="mp3")

        # Transcribe the chunk using OpenAI Whisper
        with open(temp_file_name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            full_transcription += transcript['text'] + " "

        # Delete the temporary chunk file
        os.remove(temp_file_name)

    # Delete the cache directory
    shutil.rmtree(cache_dir)

    return full_transcription


# For demonstration purposes
audio_file_path_demo = "../../290-trial-by-jury.mp3"
transcribed_text_demo = chunk_and_transcribe(audio_file_path_demo)

# Print the transcribed text
print(transcribed_text_demo)
