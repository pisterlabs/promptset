
from typing import Iterable, Iterator, List, Optional

import openai
import io

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import YoutubeAudioLoader
from langchain.document_loaders.blob_loaders import Blob, BlobLoader
from langchain.document_loaders.blob_loaders.schema import Blob
from langchain.schema import Document

from pydub import AudioSegment

import logging

import os

from math import ceil

import speech_recognition as sr

from datetime import timedelta

from utils.constants import Language

# To use secrets
import streamlit as st


AudioSegment.ffmpeg = 'ffmpeg.exe'
AudioSegment.ffprobe = 'ffprobe.exe'
AudioSegment.converter = "ffmpeg.exe"


def format_time(millis: int) -> str:
    '''
    Formats time from milliseconds to HH:MM:SS

    Args:
        millis(int):   Time in milliseconds
    Returns:
        time(str):  Time formatted as HH:MM:SS

    '''
    return str(timedelta(seconds=millis/1000))


class WhisperParser(BaseBlobParser):
    def __init__(self, api_key: str, save_dir: str, language: Optional[Language] = Language.US_English) -> None:
        self.api_key = api_key
        # Directory to save the chunks in
        self.save_dir = save_dir
        self.language = language
        self.api_key = api_key
        st.secrets["OPENAI_API_KEY"] = api_key

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        # Set the API key if provided
        if self.api_key:
            openai.api_key = self.api_key

        # Audio file from disk
        audio = AudioSegment.from_file(blob.path)

        # Define the duration of each chunk in minutes
        # Need to meet 25MB size limit for Whisper API
        chunk_duration = 20
        chunk_duration_ms = chunk_duration * 60 * 1000

        total_duration = len(audio)
        total_chunks = ceil(total_duration/chunk_duration)

        # Split the audio into chunk_duration_ms chunks
        for split_number, start_time in enumerate(range(0, len(audio), chunk_duration_ms)):
            # Audio chunk
            end_time = start_time + chunk_duration_ms
            chunk = audio[start_time: end_time]
            file_obj = io.BytesIO(chunk.export(format="mp3").read())
            if blob.source is not None:
                file_obj.name = f"{self.save_dir}/chunk{split_number}.mp3"
            else:
                file_obj.name = f"{self.save_dir}/chunk{split_number}.mp3"

            # Transcribe
            logging.debug(f"Transcribing part {split_number+1}!")
            transcript = openai.Audio.transcribe("whisper-1",
                                                 file_obj,
                                                 prompt=f"Transcribe this audio in the following language:{self.language.name}")

            yield Document(
                page_content=transcript.text,
                metadata={"source": blob.source, "chunk": split_number,
                          "start_time": start_time, "end_time": end_time,
                          'error_message': '', 'total_duration': total_duration,
                          'total_chunks': total_chunks},
            )


class SpeechRecognitionParser(BaseBlobParser):
    def __init__(self, language: Optional[Language] = Language.US_English,
                 save_dir: Optional[str] = "audio-chunks",
                 converter_path: Optional[str] = 'ffmpeg.exe') -> None:
        """
        Initializes speech recognition module

        Args:
            language: The transcribed text will be in this language.
            save_dir: The directory to save audio chunks in.
            converter_path: Path to ffmpeg converter.
        """
        self.recognizer = sr.Recognizer()
        self.language = language
        AudioSegment.converter = converter_path
        self.save_dir = save_dir

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        Returns a generator of documents

        Args:
            blob:   Blobs yielded by a DocumentLoader

        Yields:
            document: where each document contains page_content and metadata(start_time , end_time and chunk_number)
        """

        try:
            audio = AudioSegment.from_file(blob.path)

            chunk_duration = 60 * 1000  # one minute
            total_duration = len(audio)
            total_chunks = ceil(total_duration/chunk_duration)

            logging.info(
                f'Audio has a total duration of {total_duration/60000} minutes')
            logging.info(f'Audio split into {total_chunks} chunks')

            # Create a folder to save audio chunks
            if not os.path.isdir(self.save_dir):
                os.mkdir(self.save_dir)

            chunk_number = 1
            for start_time in range(0, total_duration, chunk_duration):
                end_time = start_time + chunk_duration
                chunk = audio[start_time:end_time]
                file_name = os.path.join(
                    self.save_dir, f'chunk{chunk_number}.wav')
                chunk.export(
                    file_name,
                    format='wav',
                    parameters=["-ac", "1", "-ar", "16000"]
                )

                metadata = {'start_time': start_time, 'end_time': end_time,
                            'source': blob.source, 'chunk': chunk_number, 'error_message': '',
                            'total_duration': total_duration, 'total_chunks': total_chunks}

                with sr.AudioFile(file_name) as source:
                    sound = self.recognizer.record(source)
                try:

                    text = self.recognizer.recognize_google(
                        sound, language=str(self.language.value))
                    yield Document(page_content=text, metadata=metadata)

                except sr.UnknownValueError:
                    logging.exception(
                        f"Speech Recognizer could not understand the audio from {format_time(start_time)} to {format_time(end_time)}")
                    metadata['error_message'] = f"Speech Recognizer could not understand the audio from\
                        {format_time(start_time)} to {format_time(end_time)}"

                    yield Document(page_content='', metadata=metadata)

                except sr.RequestError as e:
                    logging.exception(
                        f"Could not request results from Google Speech Recognition service;")
                    metadata['error_message'] = f"Could not request results from Google Speech Recognition service"
                    yield Document(page_content='', metadata=metadata)

                chunk_number += 1
                logging.info('Saved '+file_name)
        except:
            logging.exception('Could not load input')
            raise ValueError('Could not load input')


class AudioLoader(BlobLoader):
    def __init__(self, file_paths: List[str]) -> None:
        """
        Initialize AudioLoader with a list of file paths to load audio files from

        Args:
            file_paths: A list of file paths to load audio files from
        """
        self.file_paths = file_paths

    def yield_blobs(self) -> Iterable[Blob]:
        """
        Yield blobs from each file path provided
        """
        for path in self.file_paths:
            yield Blob.from_path(path)


class CustomYoutubeAudioLoader(YoutubeAudioLoader):
    """
    Custom Implementation of YoutubeAudioLoader to be able to load from single urls as well. 
    The current implementation of YoutubeAudioLoade uses all downloaded youtube files , 
    doesn't really support operations on single file.

    Also removes sponsor block elements if found.
    """

    def __init__(self, urls: List[str], save_dir: str):
        super().__init__(urls, save_dir)

    def get_id(self, url: str) -> str:
        """
        Extracts youtube video id from url
        """
        return url.split('?v=')[1]

    def yield_blobs(self) -> Iterable[Blob]:
        """Yield audio blobs for each url."""

        try:
            import yt_dlp
        except ImportError:
            raise ValueError(
                "yt_dlp package not found, please install it with "
                "`pip install yt_dlp`"
            )

        # Use yt_dlp to download audio given a YouTube url
        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "noplaylist": True,
            "outtmpl": self.save_dir + "/%(id)s.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                },
                {
                    'key': 'SponsorBlock',
                    'categories': ['sponsor', 'selfpromo', 'interaction', 'outro', 'intro', 'filler']
                },
                {
                    'key': 'ModifyChapters',
                    'remove_sponsor_segments': ['sponsor', 'selfpromo', 'interaction', 'intro', 'outro', 'filler'],

                }

            ],
        }

        for url in self.urls:

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download file
                ydl.download(url)

        # Get the saved file path of the downloaded video
        output_files = [
            f"{self.save_dir}/{self.get_id(url)}.m4a" for url in self.urls]

        # Yield the written blobs
        loader = AudioLoader(output_files)
        for blob in loader.yield_blobs():
            yield blob
