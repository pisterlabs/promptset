#!/usr/bin/env python3
import argparse
import logging
import os
import pathlib
import shutil
import textwrap
import time

import ffmpeg
import openai
import requests
import tenacity
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")


def format_preview_text(text: str, width: int = 80) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def parse_rate_limit_headers(response: requests.Response) -> int:
    return int(response.headers.get("x-ratelimit-reset-requests", "1")[:-1])


def before_sleep_func(retry_state: tenacity.RetryCallState):
    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        if isinstance(exception, openai._exceptions.RateLimitError):
            wait_time = parse_rate_limit_headers(exception.response)
            logging.info(
                f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying."
            )
            time.sleep(wait_time)


def text_to_speech(
    api_key: str,
    voice: str,
    text_file_path: pathlib.Path,
    path_to_save: pathlib.Path,
    silent_audio_path: pathlib.Path,
):
    try:
        if text_file_path.exists():
            with open(text_file_path, "r", encoding="utf-8") as text_file:
                text = text_file.read()
        else:
            logging.warning(f"Note file {text_file_path} does not exist.")
            return

        if not text.strip():
            logging.info(f"Text file {text_file_path} is empty. Using silent audio.")
            shutil.copy(silent_audio_path, path_to_save)
            return

        preview_text = format_preview_text(text)
        logging.info(f"\n\n{preview_text}\n\n")
        client = openai.OpenAI(api_key=api_key)
        response = client.audio.speech.create(model="tts-1", voice=voice, input=text)
        temp_path = path_to_save.with_suffix(".temp.mp3")
        with open(temp_path, "wb") as audio_file:
            audio_file.write(response.content)
            logging.info(f"Original voiceover saved to {temp_path}")

        silent_audio = ffmpeg.input(silent_audio_path)
        voiceover_audio = ffmpeg.input(temp_path)
        output_path = str(path_to_save)
        concatenated_audio = ffmpeg.concat(
            silent_audio, voiceover_audio, silent_audio, v=0, a=1
        ).output(output_path)

        ffmpeg.run(concatenated_audio)
        temp_path.unlink(missing_ok=True)

    except openai._exceptions.RateLimitError as e:
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate voiceovers from text files using OpenAI's Text-to-Speech."
    )
    parser.add_argument(
        "--voice",
        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        default="echo",
        help="Choose a voice for text-to-speech conversion. Preview voices at https://platform.openai.com/docs/guides/text-to-speech",
    )
    args = parser.parse_args()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "No OpenAI API key provided. Please set your OPENAI_API_KEY in the .env file."
        )

    notes_dir = pathlib.Path("assets/notes")
    voiceover_dir = pathlib.Path("assets/voiceovers")
    voiceover_dir.mkdir(parents=True, exist_ok=True)
    silent_audio_path = pathlib.Path("assets/silence.mp3")
    num_notes = len(list(notes_dir.glob("note_*.txt")))
    for i in range(1, num_notes + 1):
        note_file = notes_dir / f"note_{i}.txt"
        output_file = voiceover_dir / f"voiceover_{i}.mp3"
        if note_file.exists():
            text_to_speech(
                openai_api_key, args.voice, note_file, output_file, silent_audio_path
            )
        else:
            logging.warning(f"Note file {note_file} does not exist.")
