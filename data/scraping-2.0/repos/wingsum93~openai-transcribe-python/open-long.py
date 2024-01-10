"""
Break up a long recording to fit within the Whisper API's limits, with some
overlap, so no words are missed, and then feed to OpenAI Whisper API to
transcribe it to .txt file.  Written by endolith and ChatGPT-4.
"""

import openai
import math
import os
import subprocess

openai.api_key = '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'
filename = r'C:/Users/YOUR/PATH/FILE.m4a'

# Constants
max_bytes = 26214400  # From Whisper error message
overlap_seconds = 5

# Get the bit rate directly from the file
bit_rate = float(subprocess.check_output(
    ["ffprobe", "-v", "quiet", "-show_entries", "format=bit_rate", "-of",
     "default=noprint_wrappers=1:nokey=1", filename]).strip())

# Estimate the duration of each chunk
chunk_duration_s = (max_bytes * 8.0) / bit_rate * 0.9

# Get the duration of the audio file
audio_duration_s = float(subprocess.check_output(
    ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of",
     "default=noprint_wrappers=1:nokey=1", filename]).strip())

# Calculate the number of chunks
num_chunks = math.ceil(audio_duration_s / (chunk_duration_s - overlap_seconds))

transcriptions = []

output_folder = "chunks"
os.makedirs(output_folder, exist_ok=True)

# Get the file extension from the filename
file_extension = os.path.splitext(filename)[1]

for i in range(num_chunks):
    start_s = i * (chunk_duration_s - overlap_seconds)
    end_s = start_s + chunk_duration_s

    # Save the chunk to disk
    chunk_file = os.path.join(output_folder, f"chunk_{i + 1}{file_extension}")

    # Use ffmpeg to extract the chunk directly into the compressed format (m4a)
    subprocess.call(["ffmpeg", "-ss", str(start_s), "-i", filename, "-t",
                     str(chunk_duration_s), "-vn", "-acodec", "copy", "-y",
                     chunk_file])

    # Transcribe the chunk
    with open(chunk_file, "rb") as file:
        transcription = openai.Audio.transcribe("whisper-1", file)
        transcriptions.append(transcription)

# Save transcriptions to a file
with open("transcriptions.txt", "w") as file:
    for idx, transcription in enumerate(transcriptions):
        file.write(f"Chunk {idx + 1}:\n{transcription}\n\n")
