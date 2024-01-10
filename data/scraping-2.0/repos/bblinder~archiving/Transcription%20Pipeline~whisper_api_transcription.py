#!/usr/bin/env python3

"""
Downloads a video from YouTube, converts it to mp3
If it's bigger than 25MB, splits it into 10MB segments.
Transcribes it using OpenAI's Whisper API.
"""

import json
import logging
import math
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import openai
import yaml
from pydub import AudioSegment
from tqdm import tqdm
from yt_dlp import YoutubeDL

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def sanitize_filename(filename):
    """
    Remove any special characters and replace them with underscores.
    """
    import re

    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized


def download_video(url, output_dir):
    """
    Download the video at 48kbps as a mp3, and retain thumbnail and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    info = YoutubeDL().extract_info(url, download=False)
    title = sanitize_filename(info["title"])
    path_to_file = Path(output_dir) / f"{title}.mp3"

    ydl_opts = {
        "outtmpl": os.path.join(output_dir, f"{title}.%(ext)s"),
        "writethumbnail": True,
        "format": "mp3/bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegMetadata",
                "add_metadata": True,
            },
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "48",
            },
            {
                "key": "EmbedThumbnail",
            },
        ],
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # outputting the mp3 file size
    file_size_in_bytes = os.path.getsize(path_to_file)
    logger.info(f"Downloaded file size: {file_size_in_bytes / 1024 / 1024:.2f} MB")

    return path_to_file


def split_audio(path_to_file, output_dir, chunk_size_in_bytes=10 * 1024 * 1024):
    """
    Split the audio into segments of the specified size in bytes.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        title = Path(path_to_file).stem
        audio = AudioSegment.from_file(path_to_file, "mp3")
    except Exception as e:
        logger.error(f"An error occurred while reading the audio file: {e}")
        return

    # Estimate the average bitrate of the audio file
    file_size_in_bytes = os.path.getsize(path_to_file)
    duration_in_milliseconds = len(audio)
    estimated_bitrate = 8 * file_size_in_bytes / duration_in_milliseconds

    # Calculate the approximate duration of each chunk
    chunk_duration_in_milliseconds = 8 * chunk_size_in_bytes / estimated_bitrate

    num_chunks = math.ceil(duration_in_milliseconds / chunk_duration_in_milliseconds)
    logger.info(f"Splitting {title} into {num_chunks} segments.")

    for i in tqdm(range(num_chunks)):
        start = int(i * chunk_duration_in_milliseconds)
        end = int((i + 1) * chunk_duration_in_milliseconds)
        segment = audio[start:end]
        segment.export(
            Path(output_dir) / title / f"{title}_prepared_{i}.mp3", format="mp3"
        )

    # return the path to the first chunk
    return Path(output_dir) / title / f"{title}_prepared_0.mp3"


def transcribe_audio(path_to_file, output_dir, config):
    openai.api_key = config["openai_api_key"]
    os.makedirs(output_dir, exist_ok=True)
    title = Path(path_to_file).stem
    logger.info(f"Transcribing {title}... please wait... this might take a little while...")

    params = {
        "model": "whisper-1",
        "response_format": "json",
        "prompt": "Hello, please use the transcript of the previous segment if it exists.",
    }

    def transcribe_single_file(path_to_file, params):
        """Transcribe a single file."""
        try:
            with open(path_to_file, "rb") as file:
                transcript = openai.Audio.transcribe(file=file, **params)
            if not transcript:
                raise ValueError("Empty transcript received from API")
            return transcript
        except Exception as e:
            logger.error(f"An error occurred while transcribing the audio file: {e}")
            return None

    # if the file is 25MB or larger, split it into 10MB segments
    if Path(path_to_file).stat().st_size > 25 * 1024 * 1024:
        # Split the audio file into 10MB chunks using the split_audio function
        first_chunk_path = split_audio(
            path_to_file, output_dir, chunk_size_in_bytes=10 * 1024 * 1024
        )
        chunk_dir = Path(first_chunk_path).parent
        chunk_paths = sorted(chunk_dir.glob("*.mp3"))

        transcripts = []
        previous_transcript = ""

        for i, chunk_path in tqdm(enumerate(chunk_paths), total=len(chunk_paths)):
            # Append the previous transcript to the prompt for the next segment
            params["prompt"] = f"{params['prompt']}\n{previous_transcript}"
            transcript = transcribe_single_file(chunk_path, params)
            if transcript:
                transcripts.append(transcript)
                previous_transcript = transcript["text"]

        # Concatenate the transcripts
        full_transcript = {
            "text": "\n".join([t["text"] for t in transcripts]),
            "segments": [],
        }
        for t in transcripts:
            full_transcript["segments"].extend(t["segments"])
    else:
        transcript = transcribe_single_file(path_to_file, params)
        full_transcript = transcript if transcript else {"text": "", "segments": []}

    # Write the transcript to a file
    transcript_path = str(Path(output_dir) / f"{title}_transcript.json")
    with open(transcript_path, "w") as f:
        json.dump(full_transcript, f)

    # write the transcript to a text file
    text_path = str(Path(output_dir) / f"{title}_transcript.txt")
    with open(text_path, "w") as f:
        f.write(full_transcript["text"])

    return text_path


def main():
    """Putting it all together."""
    parser = ArgumentParser()
    parser.add_argument("-u", "--url", help="URL of the video to download.")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save the output files. Default is current directory.",
        default="transcription",
    )
    parser.add_argument(
        "-f", "--file", help="Path to the file to transcribe.", required=False
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to the config file.",
        default="config.yaml",
        required=True,
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.file:
        path_to_file = args.file
        logger.info(f"Transcribing {path_to_file}")
    else:
        path_to_file = download_video(args.url, args.output_dir)
        logger.info(f"Downloaded {path_to_file}")

    # check if the config file exists
    if args.config:
        if not Path(args.config).exists():
            raise ValueError(f"Config file {args.config} does not exist.")
        else:
            config_path = args.config
    else:
        config_path = "config.yaml"

    # load the config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # if the file is 25MB or larger, split it into 10 minute segments
    if Path(path_to_file).stat().st_size > 25 * 1024 * 1024:
        path_to_file = split_audio(path_to_file, args.output_dir)
        logger.info(f"Split {path_to_file}")

    transcription = str(transcribe_audio(path_to_file, args.output_dir, config))

    # ask to delete the original audio file
    delete_audio = input("Delete original audio file? (y/n): ").lower()
    if delete_audio == "y":
        os.remove(path_to_file)
        logger.info(f"Deleted {path_to_file}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
