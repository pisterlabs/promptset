"""
Lib for downloading and splitting audio.
"""

import tempfile

import youtube_dl
import openai
from pydub import AudioSegment

from .logger import logger

# Initialize OpenAI API
from .config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

class YTLogger():
    """
    Suppress youtube_dl logging bar errors.
    """
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        logger.error(msg)


def youtube_dl_to_file(url: str) -> str:
    """
    Download audio from YouTube URL using youtube_dl to a temp file.
    Returns the file path to the downloaded audio.
    """
    audio_file_name = tempfile.NamedTemporaryFile().name + ".mp3"
    ydl_opts = {
    'keepvideo': True,
    'format': 'bestaudio/best',
    'outtmpl': audio_file_name,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '128'
    }],
    'logger': YTLogger()
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return audio_file_name


def get_openai_whisper_transcript(audio_filepath: str) -> dict:
    """
    Transcribe audio file using OpenAI API.
    Returns the response from the API.
    """
    with open(audio_filepath, "rb") as audio_file:
        return openai.Audio.transcribe("whisper-1", audio_file)


class AudioHandler:
    """
    Collection of functions for handling audio.
    """
    @staticmethod
    def download_from_yt(url: str) -> str:
        """
        Provided a URL, return the path to the downloaded audio file.
        """
        audio_file_name = youtube_dl_to_file(url)
        return audio_file_name

    @staticmethod
    def split_into_chunks(filepath: str) -> list[str]:
        """
        Given an audio file, return a list of paths to the split audio files.
        """
        logger.debug(f"Splitting audio file {filepath} into chunks...")
        chunk_paths = []
        sound = AudioSegment.from_file(filepath)
        chunks = sound[::1000 * 60 * 15]
        for i, chunk in enumerate(chunks):
            audio_file_name = f"{tempfile.NamedTemporaryFile().name}_{i}.mp3"
            with open(audio_file_name, "wb") as f:
                chunk.export(f, format="mp3")
            chunk_paths.append(audio_file_name)
        return chunk_paths

    @staticmethod
    def create_transcript(audio_chunk_filenames: list) -> str:
        """
        Given a list of audio file paths, create a transcript for each file.
        Return the path to the transcript file.
        """
        transcript_filename = tempfile.NamedTemporaryFile().name + ".txt"
        with open(transcript_filename, "w", encoding="utf-8") as file:
            for filepath in audio_chunk_filenames:
                transcript = get_openai_whisper_transcript(filepath)
                file.write(str(transcript.get("text")))
        return transcript_filename
