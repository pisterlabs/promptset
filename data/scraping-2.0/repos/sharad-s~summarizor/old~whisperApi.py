import argparse
import os
import openai
import uuid
import subprocess
import logging
import time

APIKEY = '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'

# Setup logging
logging.basicConfig(level=logging.INFO)

def split_audio_into_chunks(audio_file, uuid_str):
    """Split the audio file into chunks using ffmpeg and return a list of chunk filenames."""
    logging.info(f"Splitting {audio_file} into chunks using ffmpeg...")
    start_time = time.time()

    chunk_duration_seconds = 600  # 10 minutes, you can adjust this as needed

    # Create a directory for chunks
    chunk_directory = os.path.join("tmp", uuid_str, "chunks")
    os.makedirs(chunk_directory, exist_ok=True)

    chunk_filename_template = os.path.join(chunk_directory, "chunk_%03d.mp3")
    cmd = [
        "/Users/sharadshekar/Documents/ffmpeg",
        "-i", audio_file,
        "-f", "segment",
        "-segment_time", str(chunk_duration_seconds),
        "-c", "copy",
        chunk_filename_template
    ]
    subprocess.run(cmd)

    # Return the list of generated chunks
    chunks = [os.path.join(chunk_directory, f) for f in os.listdir(chunk_directory) if f.startswith("chunk_")]
    
    logging.info(f"Finished splitting in {time.time() - start_time:.2f} seconds.")
    return sorted(chunks)


def transcribe_audio_with_openai_whisper_api(audio_file):
    """Transcribe the audio file using OpenAI's Whisper ASR API."""
    logging.info(f"Transcribing {audio_file} using Whisper API...")
    start_time = time.time()
    
    openai.api_key = APIKEY
    
    with open(audio_file, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
    
    logging.info(f"Finished transcribing in {time.time() - start_time:.2f} seconds.")
    return transcript["text"]

def summarize_text_with_openai_gpt(text):
    """Summarize the provided text using OpenAI's Chat completions API."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can also use gpt-4 if available and preferred
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Summarize the following text."},
            {"role": "user", "content": text}
        ]
    )
    
    return response['choices'][0]['message']['content']



def main():
    parser = argparse.ArgumentParser(description="Transcribe MP3 files using OpenAI's Whisper API")
    parser.add_argument("input_file", metavar="INPUT_MP3", help="Input MP3 file to be transcribed")
    args = parser.parse_args()
    
    # Generate UUID for the job
    uuid_str = str(uuid.uuid4())

    # Split the audio into manageable chunks
    chunks = split_audio_into_chunks(args.input_file, uuid_str)

    # Transcribe each chunk and combine the results
    combined_transcription = ""
    for chunk in chunks:
        transcript = transcribe_audio_with_openai_whisper_api(chunk)
        combined_transcription += transcript + "\n\n"

    # Summarize the combined transcription
    summary = summarize_text_with_openai_gpt(combined_transcription)

    # Save the combined transcription and its summary to text files
    with open(os.path.join("tmp", uuid_str, "transcription.txt"), "w") as f:
        f.write(combined_transcription)

    with open(os.path.join("tmp", uuid_str, "summary.txt"), "w") as f:
        f.write(summary)

if __name__ == "__main__":
    main()