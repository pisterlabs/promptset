import glob
import logging
import os
import subprocess

from openai import OpenAI
from openai.types.audio.transcription import Transcription
from openai.types.audio.translation import Translation

client = OpenAI()

logger = logging.getLogger(__name__)


def split_audio(file, split_sec=3600):
    logger.debug(f"Splitting {file} into {split_sec} second chunks.")

    root, ext = os.path.splitext(file)

    # Split the audio file into smaller chunks
    output = subprocess.run(
        [
            "ffmpeg",
            "-i",
            file,
            "-vn",
            "-b:a",
            "50k",
            "-f",
            "segment",
            "-segment_time",
            str(split_sec),
            f"{root}-%03d{ext}",
        ],
        capture_output=True,
    )
    if output.returncode != 0:
        logger.debug(f"FFmpeg stderr: {output.stderr.decode()}")
        logger.debug(f"FFmpeg stdout: {output.stdout.decode()}")
        raise Exception(
            f"FFmpeg failed to split {file} into {split_sec} second chunks."
        )

    # Return the list of output filenames
    files = glob.glob(f"{glob.escape(root)}-*{glob.escape(ext)}")
    logger.debug(f"Split {file} into {len(files)} files.")
    return files


def merge(transcriptions: list[Transcription | Translation]):
    logger.debug(f"Merging {len(transcriptions)} transcriptions.")

    transcription = {
        "task": transcriptions[0].task,
        "language": transcriptions[0].language,
        "duration": 0.00,
        "text": "",
        "segments": [],
    }

    id = 0
    for t in transcriptions:
        for s in t.segments:
            transcription["segments"].append(
                {
                    "id": id,
                    "seek": s["seek"] + transcription["duration"] * 100,
                    "start": s["start"] + transcription["duration"],
                    "end": s["end"] + transcription["duration"],
                    "text": s["text"],
                    "tokens": s["tokens"],
                    "temperature": s["temperature"],
                    "avg_logprob": s["avg_logprob"],
                    "compression_ratio": s["compression_ratio"],
                    "no_speech_prob": s["no_speech_prob"],
                }
            )
            id += 1

        transcription["duration"] += t.duration
        transcription["text"] += t.text

    return transcription


def transcribe_audio(file, prompt=None, language=None):
    files = [file]

    # Check if the audio file is larger than 25 MB
    if os.path.getsize(file) > 25 * 1024 * 1024:
        # Split the audio file into smaller chunks
        files = split_audio(file)

    responses = []
    for file in files:
        logger.debug(f"Transcribing {file}.")

        # Open the audio file
        with open(file, "rb") as f:
            # Transcribe the audio file using the Whisper model
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                prompt=prompt,
                response_format="verbose_json",
                language=language,
            )

        responses.append(response)

    return merge(responses)


def translate_audio(file, prompt=None):
    files = [file]

    # Check if the audio file is larger than 25 MB
    if os.path.getsize(file) > 25 * 1024 * 1024:
        # Split the audio file into smaller chunks
        files = split_audio(file)

    responses = []
    for file in files:
        logger.debug(f"Translating {file}.")

        # Open the audio file
        with open(file, "rb") as f:
            # Translate the audio file to English using the Whisper model
            response = client.audio.translations.create(
                model="whisper-1", file=f, prompt=prompt, response_format="verbose_json"
            )

        responses.append(response)

    return merge(responses)
