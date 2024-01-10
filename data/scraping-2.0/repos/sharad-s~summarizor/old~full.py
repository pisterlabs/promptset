import subprocess
import argparse
import os
import uuid
import openai
import tiktoken
import logging
import time

# Constants and Setup
APIKEY = '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'
MAX_TOKENS = 4096
SUMMARY_PREFIX_LENGTH = 150
OVERLAP_SIZE = 500
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]: %(message)s')
openai.api_key = APIKEY

enc = tiktoken.get_encoding("cl100k_base")


def tokens_in_text(text):
    return len(list(enc.encode(text)))


def split_text_into_chunks(text):
    words = text.split()
    chunks = []
    current_chunk = ""

    for word in words:
        if tokens_in_text(current_chunk + word) <= (MAX_TOKENS - SUMMARY_PREFIX_LENGTH):
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Functions from main.py


def convert_video_to_audio(input_video, output_directory):
    output_audio = os.path.join(output_directory, os.path.basename(
        os.path.splitext(input_video)[0]) + '.mp3')
    cmd = ["/Users/sharadshekar/Documents/ffmpeg", "-i",
           input_video, "-vn", "-q:a", "0", output_audio]
    subprocess.run(cmd)
    return output_audio

# Functions from whisperApi.py


def split_audio_into_chunks(audio_file, uuid_str):
    logging.info(f"Splitting {audio_file} into chunks using ffmpeg...")
    start_time = time.time()

    chunk_duration_seconds = 600
    chunk_directory = os.path.join("tmp", uuid_str, "chunks")
    os.makedirs(chunk_directory, exist_ok=True)
    chunk_filename_template = os.path.join(chunk_directory, "chunk_%03d.mp3")
    cmd = ["/Users/sharadshekar/Documents/ffmpeg", "-i", audio_file, "-f", "segment",
           "-segment_time", str(chunk_duration_seconds), "-c", "copy", chunk_filename_template]
    subprocess.run(cmd)
    chunks = [os.path.join(chunk_directory, f) for f in os.listdir(
        chunk_directory) if f.startswith("chunk_")]

    logging.info(
        f"Finished splitting in {time.time() - start_time:.2f} seconds.")
    return sorted(chunks)


def transcribe_audio_with_openai_whisper_api(audio_file):
    logging.info(f"Transcribing {audio_file} using Whisper API...")
    start_time = time.time()

    with open(audio_file, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)

    logging.info(
        f"Finished transcribing in {time.time() - start_time:.2f} seconds.")
    return transcript["text"]

# Since we're using the OpenAI Whisper API for transcription and summarization,
# we'll retain only one `summarize_text_with_openai_gpt` function.


def summarize_text_with_openai_gpt(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. You will be provided a chunk of text from audio transcription of a video. Break down the text into logical sections, then thoroughly summarize each section, capture all important points and nuances."},
            {"role": "user", "content": text}
        ]
    )
    return response['choices'][0]['message']['content']


# Main Function


# Main Function
def main():
    parser = argparse.ArgumentParser(description="Full combined process")
    parser.add_argument("input_file", metavar="INPUT_VIDEO",
                        help="Input video file to be processed")
    args = parser.parse_args()

    unique_id = str(uuid.uuid4())
    output_directory = os.path.join("out", unique_id)
    os.makedirs(output_directory, exist_ok=True)

    audio_file = convert_video_to_audio(args.input_file, output_directory)

    # Transcribe the audio
    chunks = split_audio_into_chunks(audio_file, unique_id)
    combined_transcription = ""
    for chunk in chunks:
        transcript = transcribe_audio_with_openai_whisper_api(chunk)
        combined_transcription += transcript + "\n\n"

    with open(os.path.join(output_directory, "transcription.txt"), "w") as f:
        f.write(combined_transcription)

    # Summarize
    logging.info(f"Splitting the text into chunks...")
    chunks = split_text_into_chunks(combined_transcription)

    logging.info(f"Number of chunks to be summarized: {len(chunks)}")

    summarized_chunks = []
    for idx, chunk in enumerate(chunks, 1):
        logging.info(
            f"Summarizing chunk {idx}/{len(chunks)} (Size: {tokens_in_text(chunk)} tokens)...")
        summarized_chunk = summarize_text_with_openai_gpt(chunk)
        summarized_chunks.append(summarized_chunk)

    combined_summary = summarized_chunks[0]
    for i in range(1, len(summarized_chunks)):
        current_chunk = summarized_chunks[i]
        combined_summary += ' '.join(current_chunk.split()[3:])

    output_filename = os.path.join(
        output_directory,  os.path.splitext(args.input_file)[0] + "_summary.txt")

    with open(output_filename, "w") as f:
        f.write(combined_summary)
        logging.info(f"Summary written to: {output_filename}")


if __name__ == "__main__":
    main()
