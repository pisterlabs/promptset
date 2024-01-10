import openai
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import credentials

from dotenv import load_dotenv

openai.api_key = 'sk-VkruRhbsDhZTJ8NgFM4YT3BlbkFJvzysIPK8GlZC9Qj8XQXt0'

# Get the current directory path
current_directory = os.getcwd()

# Directory where the chunks are stored
chunks_directory = os.path.join(current_directory, 'chunks')

def split_audio(audio_path, chunk_length_ms=25000):
    audio = AudioSegment.from_file(audio_path)

    chunks = make_chunks(audio, chunk_length_ms) # Make chunks of one sec

    #Export all of the individual chunks as wav files
    for i, chunk in enumerate(chunks):
        chunk_name = "chunk{0}.wav".format(i)
        print ("exporting", chunk_name)
        chunk.export(os.path.join(chunks_directory, chunk_name), format="wav")


def process_audio(audio_path):
    # Split the audio into chunks
    split_audio(audio_path)

    # Initialize an empty string to store the full transcript
    full_transcript = ""

    # Loop over each chunk
    for i in range(len(os.listdir(chunks_directory))):
        # Construct the path to the chunk
        chunk_path = os.path.join(chunks_directory, f"chunk{i}.wav")

        # Open the chunk
        with open(chunk_path, "rb") as audio_file:
            # Transcribe the chunk
            response = openai.Audio.transcribe("whisper-1", audio_file, response_format="srt")

        # Add the chunk's transcript to the full transcript
        full_transcript += response

    # Write the full transcript to a file
    with open("transcript.txt", "w", encoding='utf-8') as transcript_file:
        transcript_file.write(full_transcript)

    # Return the full transcript
    return full_transcript

