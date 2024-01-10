# -*- coding: utf-8 -*-
"""Services for the user app."""
from __future__ import unicode_literals

import logging
import os
import sys
from io import BytesIO

import dotenv
import openai
import torch
import tqdm
import whisper.transcribe
import yt_dlp as youtube_dl
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi

from riddle_me_this.oauth import get_google_token
from riddle_me_this.user.data_loading import *  # noqa: F403
from riddle_me_this.user.models import Transcript, Video

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

logging.basicConfig(
    filename="../../record.log",
    level=logging.DEBUG,
    format=f"%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s",  # noqa: F541
)


class NamedBytesIO(BytesIO):
    """An in-memory file-like object that has a name attribute."""

    def __init__(self, data, name):
        """Initialize the object with the given data and name."""
        super().__init__(data)
        self.name = name


class _CustomProgressBar(tqdm.tqdm):
    """
    Makes a custom progress bar for the transcribe_whisper function.

    https://github.com/openai/whisper/discussions/850
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n  # Set the initial value

    def update(self, n):
        """Update the progress bar."""
        super().update(n)
        self._current += n

        logging.info("Progress: " + str(self._current) + "/" + str(self.total))


def get_youtube_service():
    """
    Gets the YouTube service object using the access token obtained from the Google API.

    Returns:
    --------
    youtube : googleapiclient.discovery.Resource object
        The YouTube service object.
    """
    access_token, _ = get_google_token()
    if not access_token:
        return None
    credentials = Credentials(access_token)
    youtube = build("youtube", "v3", credentials=credentials)
    return youtube


def get_youtube_video_id(url):
    """
    Extracts the YouTube video ID from the given URL.

    Parameters:
    -----------
    url : str
        The URL of the YouTube video.

    Returns:
    --------
    video_id : str
        The ID of the YouTube video.
    Raises:
    -------
    ValueError : If the YouTube URL is not valid.
    """
    regex = re.compile(  # noqa: F405
        r"^.*(youtu\.be\/|v\/|u\/\w\/|embed\/|watch\?v=|\&v=)([^#\&\?]*).*"  # noqa
    )
    match = regex.match(url)
    if match and len(match.group(2)) == 11:
        return match.group(2)
    else:
        raise ValueError("Could not parse YouTube URL.")


def get_video_info(video_id):
    """
    Gets the video information from YouTube API and stores it in the database.

    Parameters:
    -----------
    video_id : str
        The ID of the YouTube video.

    Returns:
    --------
    video_info : Video object
        The video information object.
    Raises:
    -------
    Exception : If the YouTube service could not be created.
    """
    video_info = Video.query.filter_by(video_id=video_id).first()  # noqa
    if video_info:
        return video_info
    else:
        youtube = get_youtube_service()
        if not youtube:
            raise Exception("Failed to create YouTube service.")

        video_info = (
            youtube.videos()
            .list(part="snippet,statistics,contentDetails,status", id=video_id)
            .execute()
        )
        load_video_info(video_info)  # noqa
        return get_video_info(video_id)


def get_and_load_transcripts(video_id, language_code="en", local=True):
    """
    Searches the database for a transcript with the given video_id and language_code.

    Args:
        video_id (str): The ID of the YouTube video.
        language_code (str): The language code of the desired transcript.
        local (bool): Whether to use the local transcribe_whisper function or the remote one.

    Returns:
        transcript (Transcript or None): The transcript object if found in the database
    Raises:
    -------
    Exception : If there are no available transcripts for the given video.
    """

    # Query the database for a transcript with the given video_id and language_code
    transcript = (
        Transcript.query.filter_by(
            video_id=video_id, language_code=language_code, is_generated=False
        ).first()
        or Transcript.query.filter_by(  # noqa
            video_id=video_id, language_code="en-whisper"
        ).first()
    )
    if transcript:
        return transcript
    else:
        auto_transcript = Transcript.query.filter_by(
            video_id=video_id, language_code=language_code, is_generated=True
        ).first()
        if not auto_transcript:
            try:
                transcripts = get_transcripts(video_id)
            except Exception as e:  # noqa
                logging.error(e)  # noqa
                transcripts = [
                    {
                        "video_id": video_id,
                        "transcript": [{"text": "This is fake text"}],
                        "text": None,
                        "language_code": "en",
                        "is_generated": True,
                    }
                ]  # noqa
            load_transcripts(video_id, transcripts)  # noqa
            return get_and_load_transcripts(video_id)
        else:
            audio_file = download_audio_from_youtube(
                f"https://www.youtube.com/watch?v={video_id}"
            )
            transcripts = (
                transcribe_whisper_local(audio_file)
                if local
                else transcribe_audio_with_whisper(audio_file)
            )
            os.remove(f"/app/{audio_file.split('/')[-1]}")
            load_transcripts(video_id, transcripts)  # noqa
            return get_and_load_transcripts(
                video_id, language_code=language_code, local=local
            )


def get_transcripts(video_id):
    """
    Fetches all available transcripts for a given YouTube video and returns them as a list of transcript objects.

    Args:
        video_id (str): The ID of the YouTube video.

    Returns:
        transcripts (list): A list of transcript objects, each containing the language_code and is_generated attributes.
    """
    # Get a list of available transcripts for the video
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

    # Create a list of transcript objects
    transcripts = []

    # Loop through the transcript_list and add each transcript object to the transcripts list
    for transcript in transcript_list:
        # Get the language code and is_generated attributes for the transcript
        language_code = transcript.language_code
        is_generated = transcript.is_generated

        # Add the transcript object along with the language_code and is_generated attributes to the transcripts list
        transcripts.append(
            {
                "transcript": transcript.fetch(),
                "language_code": language_code,
                "is_generated": is_generated,
            }
        )

    # Return the list of transcripts
    return transcripts


def transcribe_audio_with_whisper(audio_file):
    """
    Transcribes the input audio file using the OpenAI Whisper ASR API.

    Parameters:
    -----------
    audio_file : str
        The location of the audio file to transcribe.

    Returns:
    --------
    transcription : list of dict
        A list containing a single transcript object.
    """
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    # Create an in-memory file object from the audio data
    audio_file = NamedBytesIO(audio_data, "audio.mp3")
    # Send the audio file to the Whisper ASR API
    response = openai.Audio.transcribe("whisper-1", audio_file)
    # Extract the transcription from the response
    transcription = response["text"].strip()
    return [
        {
            "text": transcription,
            "transcript": [{"text": transcription}],
            "language_code": "en-whisper",
            "is_generated": False,
        }
    ]


def transcribe_whisper_local(audio_location):
    """
    Transcribes the input audio file using the local Whisper transcriber.

    Parameters:
    -----------
    audio_location : str
        The location of the audio file to transcribe.

    Returns:
    --------
    transcription : list of dict
        A list containing a single transcript object.
    """
    transcribe_module = sys.modules["whisper.transcribe"]
    transcribe_module.tqdm.tqdm = _CustomProgressBar
    torch.cuda.is_available()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("tiny", device=device)
    transcription = whisper.transcribe(
        model, audio_location, fp16=False, language="English", verbose=True
    )
    return [
        {
            "text": transcription["text"],
            "transcript": transcription["segments"],
            "language_code": "en-whisper",
            "is_generated": False,
        }
    ]


def download_audio_from_youtube(url, codec="mp3", quality="64"):
    """
    Downloads the audio file of a YouTube video and returns its location.

    Parameters:
    -----------
    url : str
        The URL of the YouTube video.
    codec : str, optional
        The audio codec to use for the downloaded file. Default is 'mp3'.
    quality : str, optional
        The audio quality to use for the downloaded file. Default is '64'.

    Returns:
    --------
    audio_file : str
        The location of the downloaded audio file.
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "./%(id)s.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": codec,
                "preferredquality": quality,
            }
        ],
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        _ = ydl.extract_info(url, download=True)
    return f"./{get_youtube_video_id(url)}.{codec}"
